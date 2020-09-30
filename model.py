import os
import sys
import csv   
import json
import numpy as np
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from matplotlib.image import imread

from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input as resnet_preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout,Conv2D,MaxPooling2D
from keras.optimizers import SGD,Adam
from keras.regularizers import l1
from sklearn.model_selection import KFold
from keras.models import model_from_json
from sklearn.metrics import classification_report

target_height = 224
target_width = 224
classes_dir = './classes/'
csv_headers = ['model','activation','pretrained','type','data source','image path','ground truth','prediction']
models_present = ['basic','resnet','resnet_he_uniform','mobilenet']
pretrained_models = ['resnet', 'resnet_he_uniform', 'mobilenet']

def load_dataset():
  classes = os.listdir(classes_dir)
  train_photos, test_photos, train_labels, test_labels, train_paths, test_paths = list(), list(), list(), list(), list(), list()

  def get_encoding(class_name):
    return list(map(lambda c: 1 if c == class_name else 0, classes))

  for each in classes:
    i = 0
    for image in os.listdir(classes_dir + each):
      output = get_encoding(each)
      photo = load_img(classes_dir + each + "/" + image, target_size=(target_height, target_width))
      photo = img_to_array(photo)

      if i > 2:
        train_photos.append(photo)
        train_labels.append(output)
        train_paths.append(each + "/" + image)
      else:
        test_photos.append(photo)
        test_labels.append(output)
        test_paths.append(each + "/" + image)
      i += 1

  X_train = asarray(train_photos)
  y_train = asarray(train_labels)
  X_test = asarray(test_photos)
  y_test = asarray(test_labels)
  train_paths = asarray(train_paths)
  test_paths = asarray(test_paths)

  return classes, X_train, y_train, X_test, y_test, train_paths, test_paths

def build_basic_model():
  model = Sequential()
  model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(target_height, target_width, 3)))
  model.add(MaxPooling2D((2, 2)))
  model.add(Flatten())
  model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dropout(0.2))
  model.add(Dense(6, activation='softmax'))

  opt = SGD(lr=0.01, momentum=0.5)
  model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
  return model

def build_resnet_model(initializer=None):
  base_model = ResNet50(
  include_top=False,
  pooling='max',
  weights="imagenet",
  input_shape=(224, 224, 3)
  )
  model = Sequential()
  model.add(base_model)
  model.add(Flatten())
  model.add(Dropout(0.2))
  model.add(Dense(256, activation='relu', kernel_initializer=initializer))
  model.add(Dropout(0.2))
  model.add(Dense(6, activation='softmax', kernel_regularizer=l1(0.01)))
  model.summary()
  opt = SGD(lr=0.001, momentum=0.6)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def build_mobile_model():
  base_model = MobileNetV2(
    include_top=False,
    alpha = 0.35,
    weights="imagenet",
    pooling="max",
    input_shape=(224, 224, 3)
  )
  model = Sequential()
  model.add(base_model)
  model.add(Flatten())
  model.add(Dropout(0.5))
  model.add(Dense(256, activation='relu'))
  model.add(Dense(6, activation='softmax', kernel_regularizer=l1(0.01)))
  model.summary()
  opt = SGD(lr=0.0001, momentum=0.9)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def build_model(model_name):
  switch = {
    'basic': build_basic_model,
    'resnet': build_resnet_model,
    'resnet_he_uniform': build_resnet_model('he_uniform'),
    'mobile': build_mobile_model
  }
  fx = switch.get(model_name, lambda :'Invalid')
  return fx()    

def prepare_input(model_name, data):
  switch = {
    'resnet': resnet_preprocess_input,
    'resnet_he_uniform': resnet_preprocess_input,
    'mobile': mobilenet_preprocess_input
  }
  fx = switch.get(model_name, lambda :'Invalid')
  return fx(data)

def train_model(model, dataX, dataY, testX, testY, preprocess):
  if preprocess:
    history = model.fit(dataX, dataY, epochs=25, validation_data=(testX, testY), verbose=1)
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
    
    return model, history, acc
  else:
    datagen = ImageDataGenerator(rescale=1./255)
    testgen = ImageDataGenerator(rescale=1./255)

    history = model.fit(datagen.flow(dataX, dataY), epochs=20, validation_data=testgen.flow(testX, testY), verbose=1)
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))

    return model, history, acc

def plot_metrics(history):
  fig, (ax1, ax2) = pyplot.subplots(2)
  plot_loss(history, ax1)
  plot_accuracy(history, ax2)

def plot_loss(each, ax):
  ax.set_title('Cross Entropy Loss')
  ax.plot(each.history['loss'], color='blue', label='train')
  ax.plot(each.history['val_loss'], color='orange', label='test')

def plot_accuracy(each, ax):
  ax.set_title('Classification Accuracy')
  ax.plot(each.history['accuracy'], color='blue', label='train')
  ax.plot(each.history['val_accuracy'], color='orange', label='test')

def decode(vals):
  return list(map(lambda x: list_to_num(x), vals))

def list_to_num(ls):
  res = 1
  for i in range(len(ls)):
    if ls[i] == 1:
      res = i
      break
  return res

def get_table_metrics(model, classes, testX, testY):
  print(classification_report(decode(testY), decode(model.predict(testX)), target_names=classes))
  return json.dumps(classification_report(decode(testY), decode(model.predict(testX)), target_names=classes, output_dict=True))

def save_model(model, name):
  fd_model = os.open('./models/%s.json' % name, os.O_RDWR|os.O_CREAT) 
  model_json = model.to_json()
  os.write(fd_model, bytes(model_json, 'utf-8'))
  os.close(fd_model)

  model.save_weights("./weights/%s.h5" % name)
  print("Saved model to disk")

def write_to_output(model, name, classes, testX, testY, test_paths, trainX, trainY, train_paths):
  with open('./csv_output/%s.csv' % name, mode='w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(csv_headers)
    
    for i in range(len(testX)):
      writer.writerow([name, 'softmax', name in pretrained_models, 'test', 'classes', str(test_paths[i]), 
                       classes[list_to_num(testY[i])], classes[decode(model.predict(testX[i:i+1]))[0]]])
    for i in range(len(trainX)):
      writer.writerow([name, 'softmax', name in pretrained_models, 'train', 'classes', str(train_paths[i]), 
                    classes[list_to_num(trainY[i])], classes[decode(model.predict(trainX[i:i+1]))[0]]])

  print('Written to output CSV')

def run_classification(model_name):
  preprocess = model_name in pretrained_models

  classes, trainX, trainY, testX, testY, train_paths, test_paths = load_dataset()
  model = build_model(model_name)

  if preprocess:
    trainX = prepare_input(model_name, trainX)
    testX = prepare_input(model_name, testX)

  model, hists, accs = train_model(model, trainX, trainY, testX, testY, preprocess)

  plot_metrics(hists)
  report = get_table_metrics(model, classes, testX, testY)

  save_model(model, model_name)
  write_to_output(model, model_name, classes, testX, testY, test_paths, trainX, trainY, train_paths)
  
if len(sys.argv) != 2:
  print('Please specify an argument for the model name')
else:
  if sys.argv[1] in models_present:
    print('Model found')
    run_classification(sys.argv[1])
  else:
    print('Model not found')
  