# -*- coding: utf-8 -*-
"""
reconhecimento.ipynb
Original file is located at
https://colab.research.google.com/drive/1qO1xb4pq_-gpVVMizfqi-3x1hbf9GsCi
"""

import cv2
import numpy as np
from PIL import Image
from google.colab.patches import cv2_imshow
import os
import zipfile

#local do arquivo zip
path = '/content/yalefaces.zip'
#extraindo arquivo zip
zip_object = zipfile.ZipFile(file=path, mode = 'r')
zip_object.extractall('./')
zip_object.close()

#listando o diretorio para ver o que tem dentro
print(os.listdir('/content/yalefaces/train'))

#pegando imagem de referencia
imagem_teste =  '/content/yalefaces/train/subject01.leftlight.gif'
imagem = Image.open(imagem_teste).convert('L')

#mostrando imagem
imagem

#mostrando o tamanho da imagem
imagem_np = np.array(imagem, 'uint8')
cv2_imshow(imagem_np)
print(imagem_np.shape)

"""Detecção Facial com LBPH e OpenCV"""

#adicionando a rede neural pre treinada pelo openCV
network = cv2.dnn.readNetFromCaffe('/content/deploy.prototxt.txt', '/content/res10_300x300_ssd_iter_140000.caffemodel')

#mudando a escala de cor e o formato da imagem
imagem = cv2.cvtColor(imagem_np, cv2.COLOR_GRAY2BGR)
(h, w) = imagem.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(imagem, (100, 100)), 1.0, (100,100), (104.0, 117.0, 123.0))
network.setInput(blob)
deteccoes = network.forward()

#adicionando a linha de detecção do rosto, porcentagem de acurácia e cor
conf_min = 0.7
imagem_cp = imagem.copy()
for i in range(0, deteccoes.shape[2]):
  confianca = deteccoes[0, 0, i, 2]
  if confianca > conf_min:
    bbox = deteccoes[0, 0, i, 3:7] * np.array([w, h, w, h])
    (start_x, start_y, end_x, end_y) = bbox.astype('int')
    roi = imagem_cp[start_y:end_y, start_x:end_x]
    text = "{:.2f}%".format(confianca * 100)
    cv2.putText(imagem, text, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 2)
    cv2.rectangle(imagem, (start_x, start_y), (end_x, end_y), (0,255,0), 2)
face = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
cv2_imshow(imagem)
print(imagem.shape)

#recortando a parte dtectada do rosto
face = cv2.resize(face, (60,80))
cv2_imshow(face)
print(face.shape)

#criando um metodo que detecta a face
def detecta_face(network, path_imagem, conf_min = 0.7):

  #leitura e processamento da imagem
  imagem = Image.open(path_imagem).convert('L')
  imagem = np.array(imagem, 'uint8')
  imagem = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
  (h, w) = imagem.shape[:2]

  #preparando a imagem para a rede neural
  blob = cv2.dnn.blobFromImage(cv2.resize(imagem, (100, 100)), 1.0, (100,100), (104.0, 117.0, 123.0))
  network.setInput(blob)

  #obtendo a detecção
  deteccoes = network.forward()

  #processando a detecção
  face = None
  for i in range(0, deteccoes.shape[2]):
    confianca = deteccoes[0, 0, i, 2]
    if confianca > conf_min:
      bbox = deteccoes[0, 0, i, 3:7] * np.array([w, h, w, h])
      (start_x, start_y, end_x, end_y) = bbox.astype('int')
      roi = imagem[start_y:end_y, start_x:end_x]
      roi = cv2.resize(roi, (60,80))
      cv2.rectangle(imagem, (start_x, start_y), (end_x, end_y), (0,255,0), 2)
      face = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

  #retorno
  return face, imagem

#pega imagem de teste
teste_imagem = '/content/yalefaces/train/subject01.sad.gif'

#separa o rosto e a imagem completa para receber os dois separadamente
face, imagem = detecta_face(network, teste_imagem)
cv2_imshow(imagem)
cv2_imshow(face)

"""Construção da base, além de fazer o reconhecimento iremos fazer a partição dessa imagem, pegando um codigo por imagem. criando um ID ou rotulo no nome da imagem substituindo o nome da imagem."""

#essa função carrega imagens para o treinamento
def get_image_data():
  paths = [os.path.join('/content/yalefaces/train', f) for f in os.listdir('/content/yalefaces/train')]

  #Inicialização das listas para armazenar os rostos e IDs
  faces = []
  ids = []

  #Processamento de cada imagem
  for path in paths:
    face, imagem = detecta_face(network, path)
    #cv2_imshow(imagem)
    #cv2_imshow(face)
    #print(path)
    id = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))
    #print(id)
    ids.append(id)
    faces.append(face)
    cv2_imshow(face)

  #retorno
  return np.array(ids), faces

ids, faces = get_image_data()

"""Eigenfaces"""

#criação e treinamento do classificador
eigen_classifier = cv2.face.EigenFaceRecognizer_create()
eigen_classifier.train(faces, ids)
eigen_classifier.write('/content/eigen_classifier.yml')

#eitura do classificador treinado
eigen_classifier = cv2.face.EigenFaceRecognizer_create()
eigen_classifier.read('/content/eigen_classifier.yml')

#Teste da detecção de rosto em uma nova imagem
imagem_teste = '/content/yalefaces/test/subject03.glasses.gif'
face, imagem = detecta_face(network, imagem_teste)
face, face.shape

#usando previsao
previsao = eigen_classifier.predict(face)
previsao

#saida
saida_esperada = int(os.path.split(imagem_teste)[1].split('.')[0].replace('subject', ''))
saida_esperada

#juntando a predição com a saida esprada e adicionando na imagem
cv2.putText(imagem, 'Pred: ' + str(previsao[0]), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0))
cv2.putText(imagem, 'Exp: ' + str(saida_esperada), (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0))
cv2_imshow(imagem)

#teste de desempenho do classificador
def teste_reconhecimento(imagem_teste, classificador, show_conf = False):
  face, imagem_np = detecta_face(network, imagem_teste)
  previsao, conf = classificador.predict(face)
  saida_esperada = int(os.path.split(imagem_teste)[1].split('.')[0].replace('subject', ''))
  cv2.putText(imagem_np, 'Pred: ' + str(previsao), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0))
  cv2.putText(imagem_np, 'Exp: ' + str(saida_esperada), (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0))
  if show_conf:
    print(conf)
  return imagem_np, previsao

imagem_teste = '/content/yalefaces/test/subject11.happy.gif'
imagem_np, previsao = teste_reconhecimento(imagem_teste, eigen_classifier, True)
cv2_imshow(imagem_np)

"""Avaliação do algoritimo"""

def avalia_algoritmo(paths, classificador):
  previsoes = []
  saidas_esperadas = []
  for path in paths:
    face, imagem = detecta_face(network, path)
    previsao, conf = classificador.predict(face)
    saida_esperada = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))
    previsoes.append(previsao)
    saidas_esperadas.append(saida_esperada)
  return np.array(previsoes), np.array(saidas_esperadas)

paths_teste = [os.path.join('/content/yalefaces/test', f) for f in os.listdir('/content/yalefaces/test')]
print(paths_teste)

previsoes, saidas_esperadas = avalia_algoritmo(paths_teste, eigen_classifier)
previsoes

saidas_esperadas

from sklearn.metrics import accuracy_score
accuracy_score(saidas_esperadas, previsoes)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(saidas_esperadas, previsoes)
cm

import seaborn
seaborn.heatmap(cm, annot=True);

"""FicherFaces"""

fisher_classifier = cv2.face.FisherFaceRecognizer_create()
fisher_classifier.train(faces, ids)
fisher_classifier.write('fisher_classifier.yml')

fisher_classifier = cv2.face.FisherFaceRecognizer_create()
fisher_classifier.read('/content/fisher_classifier.yml')

imagem_teste = '/content/yalefaces/test/subject07.happy.gif'
imagem_np, previsao = teste_reconhecimento(imagem_teste, fisher_classifier, True)
cv2_imshow(imagem_np)

"""Avaliacao do algoritimo"""

resultados_avaliacao = (paths_teste, fisher_classifier)

"""parametros ficherfaces"""

fisher_classifier_2 = cv2.face.FisherFaceRecognizer_create(5)
fisher_classifier_2.train(faces, ids)
imagem_teste = '/content/yalefaces/test/subject07.happy.gif'
imagem_np, previsao = teste_reconhecimento(imagem_teste, fisher_classifier_2, True)
cv2_imshow(imagem_np)

fisher_classifier_2 = cv2.face.FisherFaceRecognizer_create(20, 1000)
fisher_classifier_2.train(faces, ids)
imagem_teste = '/content/yalefaces/test/subject07.happy.gif'
imagem_np, previsao = teste_reconhecimento(imagem_teste, fisher_classifier_2, True)
cv2_imshow(imagem_np)

resultados_avaliacao = (paths_teste, fisher_classifier_2)

bph_classifier = cv2.face.LBPHFaceRecognizer_create()
bph_classifier.train (faces, ids)
bph_classifier.write ('lbph_classifier.yml')

bph_classifier = cv2.face.LBPHFaceRecognizer_create()
bph_classifier.read('/content/lbph_classifier.yml')

imagem_teste = '/content/yalefaces/test/subject01.gif'
imagem_np, previsao = teste_reconhecimento(imagem_teste, bph_classifier, True)
cv2_imshow(imagem_np)

resultados_avaliacao = (paths_teste, bph_classifier)

lbph_classifier_2 = cv2.face.LBPHFaceRecognizer_create(radius = 4)
lbph_classifier_2.train(faces, ids)
imagem_np, previsao = teste_reconhecimento(imagem_teste, lbph_classifier_2, True)
cv2_imshow(imagem_np)

lbph_classifier_2 = cv2.face.LBPHFaceRecognizer_create(radius = 4, neighbors = 12, grid_x=14, grid_y=14)
lbph_classifier_2.train(faces, ids)
imagem_np, previsao = teste_reconhecimento(imagem_teste, lbph_classifier_2, True)
cv2_imshow(imagem_np)

lbph_classifier_2 = cv2.face.LBPHFaceRecognizer_create(radius = 4, neighbors = 12, grid_x=14, grid_y=14, threshold = 550)
lbph_classifier_2.train(faces, ids)
imagem_np, previsao = teste_reconhecimento(imagem_teste, lbph_classifier_2, True)
cv2_imshow(imagem_np)