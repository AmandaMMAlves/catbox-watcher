from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn import metrics

def get_target_data_tf(y_train, y_test, classes):
    #Usar o labelEncoder para transformar nossas labels em numeros binarios
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    #Converte um vetor de classe (inteiros) em uma matriz de classe categorica
    y_train_tf = keras.utils.to_categorical(y_train_encoded, len(classes))
    y_test_tf = keras.utils.to_categorical(y_test_encoded, len(classes))
    
    return y_train_tf, y_test_tf

def get_cnn_model(y_train_tf):
    input_shape = (64, 64, 3) # imagens em dimensões 64x64 em RGB

    learning_rate = 0.001

    tf.random.set_seed(42) #Define uma semente aleatória para utilizar sempre os mesmo dados durante nossos testes
    model = Sequential() #Abrindo uma sequencia de modelo

    #1 camada convolucional com 128 neuronios
    #filtro utilizando uma matriz 3x3 tendo o deslocamento de 2
    #padding com bordas zero; função de ativação RELU; regularização L2 ativada (suavizar a penalidade dos coeficientes)
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', input_shape=input_shape))

    model.add(BatchNormalization()) #Camada de normalização para estabilizar o treinamento e acelerar convergencia

    #Camada MaxPolling ativada, com uma matriz 2x2. Padding ativado
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

    #Regularizacao Dropout ativada
    model.add(Dropout(0.2)) #remove 20%

    #2 camada convolucional com 92 neuronios
    #filtro utilizando uma matriz 3x3 tendo o deslocamento de 2
    #padding com bordas zero; função de ativação RELU; regularização L2 ativada (suavizar a penalidade dos coeficientes)
    model.add(Conv2D(92, (3,3), strides=(2,2), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
    model.add(BatchNormalization())
    #Camada MaxPolling ativada, com uma matriz 2x2. Padding ativado
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    #Regularizacao Dropout ativada
    model.add(Dropout(0.2)) #remove 20%

    #Camada que achata os dados da imagem
    model.add(Flatten())
    #Camada Dense da rede neural convolucional + função ativação ReLU
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))  # Aumentado dropout para regularização

    #Camada de saida da rede, utilizando a função de ativação sigmoid, classificação binaria
    model.add(Dense(y_train_tf.shape[1], activation='sigmoid'))

    #sigmoid -> classificação binária
    #softmax -> classificação multiclasses

    #y_train_tf = 2

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])

    model.save('modelo/my_model.keras')

    model.summary()
    return model

def get_monitor():
    return EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10, verbose=1, mode='auto')

"""Vamos configurar agora como próximo passo o ModelCheckpoint para usar os melhores pesos 
para este modelo. Um dos benefícios do ModelCheckpoint é salvar uma cópia do modelo em disco 
em intervalos regulares (como após cada época de processamento) para que você possa retomar 
o treinamento a partir do ponto em que parou, minimizando perdas de tempo e recursos computacionais"""
def get_checkpointer():
    return ModelCheckpoint(filepath='modelo/weights.best.hdf5.keras', verbose=0, save_best_only=True)

def show_val_loss_history(history):
    # %%Avaliando o modelo
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower left')
    plt.grid(True)
    plt.show()

def show_val_accuracy_history(history):
    #Validação Acuracia por acurácia por epocas
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower left')
    plt.grid(True)
    plt.show()

def show_metrics(y_true, cnn_predict, classes):
    print("Resultados: \n")
    cnn_cm = metrics.confusion_matrix(y_true, cnn_predict)

    cnn_accuracy = metrics.classification_report(y_true, cnn_predict, target_names=classes)

    print(metrics.classification_report(y_true, cnn_predict))

    roc_auc = metrics.roc_auc_score(y_true, cnn_predict)
    print(f"ROC AUC: {roc_auc}")

def get_precision(y_true, cnn_predict):
    return metrics.precision_score(y_true, cnn_predict)

def get_recall(y_true, cnn_predict):
    return metrics.recall_score(y_true, cnn_predict)

def get_f1(y_true, cnn_predict):
    return metrics.f1_score(y_true, cnn_predict)

def get_roc_auc(y_true, cnn_predict):
    return metrics.roc_auc_score(y_true, cnn_predict)