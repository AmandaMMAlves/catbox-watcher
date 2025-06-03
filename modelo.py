from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

import utils.obtencaoMassaUtils as om
import utils.modeloUtils as mu

# %%Obtendo o dataset

caminho_pasta = "dataset/"
classes = ['cat-in-the-box', 'cat-out-the-box']

pasta_imgs_cat_in = Path(caminho_pasta + classes[0])
pasta_imgs_cat_out = Path(caminho_pasta + classes[1])

extensao = ".jpg"

#Vou renomear todos os itens para que todas as imagens sejam .jpg
om.renomear_arquivos(pasta_imgs_cat_in, extensao)
om.renomear_arquivos(pasta_imgs_cat_out, extensao)

# om.exibir_numero_itens_filepath(pasta_imgs_cortantes, pasta_imgs_nao_cortantes)

print("Realizando balanceamento manual...\n\n")
filepaths_cat_in, filepaths_cat_out = om.manual_balance(pasta_imgs_cat_in, pasta_imgs_cat_out)


print("Obtendo rótulo das imagens...\n\n")
labels_cat_in, images_cat_in = om.get_labels_images(filepaths_cat_in)
labels_cat_out, images_cat_out = om.get_labels_images(filepaths_cat_out)
labels = np.concatenate((labels_cat_in, labels_cat_out))
images = np.concatenate((images_cat_in, images_cat_out))

print("Quantidade de imagens e rótulos obtidos:")
print(len(images))
print(len(labels))

print("Imagens e rótulos obtidos com sucesso!\n\n")


x_train, x_test, y_train, y_test = om.get_train_test_data(filepaths_cat_in, filepaths_cat_out, labels, images)

print("Base de treino e teste separadas com sucesso!\n\n")

# %%Preparando os dados para a rede convolucional
print("Preparando dados para a rede convolucional... \n\n")
#Achatando os dados de teste e treino
x_train_flat = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3]) #TODO: Só é usado aqui, depois testa utilizando ele para treinar
x_test_flat = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3]) #TODO: Só é usado aqui, depois testa utilizando ele para treinar

#Normalizando os dados
y_train_tf, y_test_tf = mu.get_target_data_tf(y_train, y_test, classes)


checkpointer = mu.get_checkpointer()
monitor = mu.get_monitor()
model = mu.get_cnn_model(y_train_tf)

# #Treinamento do model
print("Treinando o modelo com a base balanceada tratada... \n\n")
history = model.fit(x_train, y_train_tf, validation_split=0.25, callbacks=[monitor,checkpointer],
                    verbose=1, epochs=45, batch_size=50, shuffle=True)

# mu.show_val_loss_history(history)

# mu.show_val_accuracy_history(history)

# %%Testando o modelo
cnn_predict = model.predict(x_test)

cnn_predict = np.argmax(cnn_predict, axis=1)

y_true = np.argmax(y_test_tf, axis=1)

mu.show_metrics(y_true, cnn_predict, classes) #EXIBE RESULTADOS FINAIS


print("Treinando uma nova instancia do modelo com a abordagem de validação cruzada:\n\n")

# Configuração para validação cruzada
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

fold_model = mu.get_cnn_model(y_train_tf)
fold_accuracies = []
fold_precision = []
fold_recall = []
fold_roc_auc = []
fold_f1 = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train, y_train)):
    print(f"Treinando o fold {fold + 1}...")
    
    # Dados de treino e validação
    x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
    y_train_fold, y_val_fold = y_train_tf[train_idx], y_train_tf[val_idx]
    
    # Callbacks
    checkpointer = mu.get_checkpointer()
    monitor = mu.get_monitor()

    # Treinamento
    history = fold_model.fit(
        x_train_fold, y_train_fold,
        validation_data=(x_val_fold, y_val_fold),
        epochs=50,
        callbacks=[checkpointer, monitor],
        verbose=1
    )
    # Avaliação
    predict_value = fold_model.predict(x_val_fold)
    predict_classes = np.argmax(predict_value, axis=1)
    true_classes = np.argmax(y_val_fold, axis=1)

    val_loss, val_accuracy = fold_model.evaluate(x_val_fold, y_val_fold, verbose=0)
    fold_accuracies.append(val_accuracy)
    fold_precision.append(mu.get_precision(true_classes, predict_classes))
    fold_recall.append(mu.get_recall(true_classes, predict_classes))
    fold_roc_auc.append(mu.get_roc_auc(true_classes, predict_classes))
    fold_f1.append(mu.get_f1(true_classes, predict_classes))


# Desempenho médio
print(f"Acurácia média da validação cruzada: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
print(f"Precisão média da validação cruzada: {np.mean(fold_precision):.4f} (+/- {np.std(fold_precision):.4f})")
print(f"Recall médio da validação cruzada: {np.mean(fold_recall):.4f} (+/- {np.std(fold_recall):.4f})")
print(f"ROC AUC médio da validação cruzada: {np.mean(fold_roc_auc):.4f} (+/- {np.std(fold_roc_auc):.4f})")  
print(f"F1 médio da validação cruzada: {np.mean(fold_f1):.4f} (+/- {np.std(fold_f1):.4f})")


cnn_folder_predict = fold_model.predict(x_test)

cnn_folder_predict = np.argmax(cnn_folder_predict, axis=1)

mu.show_metrics(y_true, cnn_folder_predict, classes) #EXIBE RESULTADOS FINAIS