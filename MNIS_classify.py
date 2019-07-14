"""
Handwritten digit recognition in MNIST database using deep CNN
This code does:
             Load MNISTS dataset (60k training +10k test images)
             Creat batches of Train and Validation data (test samples are given by MNIST)
             Creat a CNN model
             Train CNN
             Evaluate CNN performance in classification by reporting loss and accuracy
 Network input: Images (28x28x1) of digits 0,1,...,9
 Network output: Class of image ("0", "1", ..., "9")
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.model_selection import train_test_split
import seaborn as sns # for statistical visualization

# Params
image_w = 28 # width
image_h = 28 # height
num_class = 10 # digit classes in MNIST
display = True # enable/disable showing plots
valid_ratio = 0.25 # validation samples selected from training images


# Loading MNIST
mnist = keras.datasets.mnist;
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# Reshaping and normalizing the images
x_train = x_train / 255.0 # normalizing image data
x_test = x_test / 255.0 
x_train.reshape(x_train.shape[0], 28, 28,1);
x_test.reshape(x_test.shape[0],28,28,1);

# Batch generation
(train_images, train_labels) = skl.utils.shuffle(x_train,y_train) # shuffling
train_images, valid_images, train_labels, valid_labels= train_test_split(train_images, train_labels, test_size=0.2, random_state=42) # split train data into train & validation data
test_images = x_test
test_labels = y_test
train_size = len(train_images)
valid_size = len(valid_images)
test_size = len(test_images)
print('train size:',len(train_labels), ' validation size:', len(valid_labels), ' test size:', len(test_labels))

#=== Showing some statistics on database
if display:
  # Show histogram of batches
  fig, 
  ax = plt.subplot(1,3,1); sns.countplot(train_labels); ax.set_title('train data hist'); 
  ax = plt.subplot(1,3,2); sns.countplot(valid_labels); ax.set_title('valid data his');
  ax = plt.subplot(1,3,3); sns.countplot(test_labels); ax.set_title('test data hist');

#=== Preparing data for network
train_images = train_images.reshape((len(train_labels), 28, 28, 1)) # for CNN inputs
valid_images = valid_images.reshape(len(valid_labels),28,28,1)
test_images = test_images.reshape(len(test_labels),28,28,1)

#=== Creating the deep network model
model = tf.keras.models.Sequential([
  keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=(28,28,1)),
  keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'),
  keras.layers.MaxPool2D((2, 2)),
  keras.layers.Dropout(0.20),
  keras.layers.Conv2D(32, (3, 3), activation='relu',kernel_initializer='he_normal'),
  keras.layers.Conv2D(32, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'),
  keras.layers.MaxPool2D(pool_size=(2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(512, activation=tf.nn.relu),
  keras.layers.Dense(64, activation=tf.nn.relu),
  keras.layers.BatchNormalization(),
  keras.layers.Dense(num_class, activation=tf.nn.softmax)
])

#=== Compiling and fitting the model to data
model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
h = model.fit(train_images,train_labels,epochs=5, verbose=1, validation_data=(valid_images,valid_labels))

#=== Evaluating the trained network
print(model.summary())
train_loss, train_acc = model.evaluate(train_images, train_labels)
valid_loss, valid_acc = model.evaluate(valid_images, valid_labels)
test_loss, test_acc = model.evaluate(test_images, test_labels)

#=== Reporting the performance measures
print('Accuracy:  train:', format(train_acc,'.3f'), ' valid:', format(valid_acc,'.3f'), ' test:', format(test_acc,'.3f'))
print('    Loss:  train:', format(train_loss,'.3f'), ' valid:', format(valid_loss,'.3f'), ' test:', format(test_loss,'.3f'))