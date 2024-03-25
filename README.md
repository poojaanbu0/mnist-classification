# Convolutional Deep Neural Network for Digit Classification

## AIM
To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
```
Problem Statement: Handwritten Digit Recognition with Convolutional Neural Networks

Objective: Develop a Convolutional Neural Network (CNN) model to accurately classify handwritten digits (0-9) from the MNIST dataset.

Data: The MNIST dataset, a widely used benchmark for image classification, contains grayscale images of handwritten digits (28x28 pixels). Each image is labeled with the corresponding digit (0-9).
```

## Neural Network Model
![Screenshot 2024-03-24 230510](https://github.com/poojaanbu0/mnist-classification/assets/119390329/d07f1f5d-484d-4f74-9caa-00a63e45ed4b)

## DESIGN STEPS

### STEP 1:

### STEP 2:

### STEP 3:


## PROGRAM

### Name: Pooja A
### Register Number: 212222240072
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape

X_test.shape

single_image= X_train[0]

single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape = (28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation = 'relu'))
model.add(layers.Dense(10,activation = 'softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)

print("Pooja A")
print("212222240072")
metrics.head()

print("Pooja A")
print("212222240072")
metrics[['accuracy','val_accuracy']].plot()

print("Pooja A")
print("212222240072")
metrics[['loss','val_loss']].plot()

print("Pooja A")
print("212222240072")
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print("Pooja A")
print("212222240072")
print(confusion_matrix(y_test,x_test_predictions))

print("Pooja A")
print("212222240072")
print(classification_report(y_test,x_test_predictions))

img = image.load_img('/content/images.png')

type(img)

img = image.load_img('/content/images.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

print("Pooja A")
print("212222240072")
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print("Pooja A")
print("212222240082")
print(x_single_prediction)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/poojaanbu0/mnist-classification/assets/119390329/fe38ee74-79b7-4794-be2c-2f48bc8fd9a0)

![image](https://github.com/poojaanbu0/mnist-classification/assets/119390329/a91d9f51-2d1f-4289-a68a-3efb74e48c36)

![image](https://github.com/poojaanbu0/mnist-classification/assets/119390329/238ddb5d-e30f-437c-b9b3-de76c6c38687)

### Classification Report
![image](https://github.com/poojaanbu0/mnist-classification/assets/119390329/b9112c73-abee-4579-90ed-61548de88b94)

### Confusion Matrix
![image](https://github.com/poojaanbu0/mnist-classification/assets/119390329/515d3c77-3c87-482a-b49b-acea76c33453)

### New Sample Data Prediction

### Input:
![images](https://github.com/poojaanbu0/mnist-classification/assets/119390329/0ab9978b-8409-43f2-9118-2ce3ad4d8dbe)

### Output:
![image](https://github.com/poojaanbu0/mnist-classification/assets/119390329/eeccefd3-5b2c-40ae-a6c0-13a13b1ed79a)


## RESULT
Thus a convolutional deep neural network for digit classification is developed and the response for scanned handwritten images is verified.
