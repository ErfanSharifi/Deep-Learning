import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import numpy as np


fashion = fashion_mnist.load_data()

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Explore Data

print ('Train Images',train_images.shape)
print ('Train Labels',train_labels.shape)
print ('Test Image:', test_images.shape)
print ('Test Labels:',test_labels.shape) 


#Preparing Data

# # plt.figure()
# # plt.imshow(train_images[5000],cmap='gray')
# # plt.colorbar()
# # plt.show()


#عکس ها رو در بازه ۰-۱ نمایش میدهد چون ماشین میتواند تشخیص دهد کجا رنگ سفید است کجا سیاه.این به این خاطر است که سیستم را ما خاکستری کردیم تا نقاط مهم که سفید هستند و خود شکل هستند را نشان بدهد

train_images = train_images / 255.0
test_images = test_images / 255.0

# # plt.figure()
# # plt.imshow(train_images[5000],cmap='gray')
# # plt.colorbar()
# # plt.show()


#Show SOme Picture
#To verify that the data is in the correct format and that you're ready to build and train the network, let's display the first 25 images from the training set and display the class name below each image.

# # plt.figure(figsize=(7,7))
# # for i in range(25):
# #     plt.subplot(5,5,i+1)
# #     plt.xticks([])
# #     plt.yticks([])
# #     plt.grid(False)
# #     plt.imshow(train_images[i], cmap = plt.cm.binary)
# #     plt.imshow(train_images[i], cmap = 'gray')
# #     plt.xlabel(class_names[train_labels[i]])
# # plt.show()


#Normalize
#One hot


# # x_train = np.array(train_images)
# # y_train = np_utils.to_categorical(train_labels, num_classes=10)


# # x_test = np.array((test_images))
# # y_test = np_utils.to_categorical(test_labels, num_classes=10)

# # x_train = x_train.astype('float32')
# # x_test = x_test.astype('float32')
# # x_train /= 255
# # x_test /= 255


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.summary()



model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=20)

#Evaluate accuracy

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
