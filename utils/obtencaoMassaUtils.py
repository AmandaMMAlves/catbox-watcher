from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import cv2

# %%Obtendo o dataset

#Vou renomear todos os itens para que todas as imagens sejam .jpg
def renomear_arquivos(pasta_imgs, extensao):
    for arquivo in pasta_imgs.iterdir():
        if arquivo.is_file():
            novo_nome = arquivo.with_suffix(extensao)
            arquivo.rename(novo_nome)

    print(f"Arquivos na pasta {pasta_imgs} foram renomeados para a extensão {extensao}.\n\n")


def exibir_numero_itens_filepath(pasta_imgs_cat_in, pasta_imgs_cat_out):
    filepaths_cat_in = get_filepaths(pasta_imgs_cat_in)
    filepaths_cat_out = get_filepaths(pasta_imgs_cat_out)

    print("Pasta de gatos dentro da caixa:")
    print(len(filepaths_cat_in))
    print("Pasta de gatos fora da caixa:")
    print(f"{len(filepaths_cat_out)} \n\n")

def get_filepaths(pasta_imgs):
    return list(pasta_imgs.glob(r'**/*.jpg'))



# %%Balanceando manualmente o dataset
#Vou pegar a quantidade de imagens da pasta com menos imagens e vou pegar a mesma quantidade da outra pasta
def manual_balance(pasta_imgs_cat_in, pasta_imgs_cat_out):
    filepaths_cat_in = get_filepaths(pasta_imgs_cat_in)
    filepaths_cat_out = get_filepaths(pasta_imgs_cat_out)
    
    minimo = min(len(filepaths_cat_in), len(filepaths_cat_out))
    filepaths_cat_in = filepaths_cat_in[:minimo]
    filepaths_cat_out = filepaths_cat_out[:minimo]
    return filepaths_cat_in, filepaths_cat_out

def get_labels_images(filepaths):
    labels = []
    images = []

    image_size = 64

    for filepath in filepaths:
        head = os.path.split(filepath)
        obj = os.path.split(head[0])

        labels.append(obj[1])#armazena rotulo cat_in ou cat_out obtido pelo path

        img = cv2.imread(str(filepath))

        # Check if image was loaded correctly
        if img is not None:
            img = cv2.resize(img, (image_size, image_size)).astype('float32') / 255.0 #redimensiona e normaliza imagem
            images.append(img)
        else:
            print(f"Failed to load image: {filepath}") # Print the problematic filepath

    #Converte a imagem em lista de array
    images = np.array(images)
    #Converte as labels para uma lista de array
    labels = np.array(labels)
    
    return labels, images


def get_train_test_data(filepaths_cat_in, filepaths_cat_out, labels, images):
    # %%Construindo o dataframe
    print("Iniciando a construção do dataframe... \n\n")
    filepaths = filepaths_cat_in + filepaths_cat_out
    pd_filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    pd_labels = pd.Series(labels, name='Label')

    df = pd.concat([pd_filepaths, pd_labels], axis=1)
    print(df['Label'].value_counts())

    # %%Embaralhando o dataset
    print("Embaralhando o dataset e exibindo as primeiras linhas... \n\n")
    df = df.sample(frac=1).reset_index(drop=True)
    print(df.head())
    print(df['Label'].value_counts())
    print("Dataset embaralhado com sucesso!\n\n")

    # %%Separando a base de treino e teste
    print("Separando base de 80% treino e de 20% teste... \n\n")
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)
    return  x_train, x_test, y_train, y_test
