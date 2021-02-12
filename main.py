import cv2 as cv
import numpy as num
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as mat
import tensorflow as tf
import pandas as pd

#############
# data test
data = pd.read_csv("C:\\Users\\FFaye\\OneDrive\\Desktop\\archive\\emnist-byclass-test.csv")
data_test = data.copy()
test_lable = data_test.values[:, 0]
test_letter = data_test.values[:, 1:]

# data train
data_1 = pd.read_csv("C:\\Users\\FFaye\\OneDrive\\Desktop\\archive\\emnist-byclass-train.csv")
data_train = data_1.copy()
train_lable = data_train.values[:, 0]
train_letter = data_train.values[:, 1:]

# preprocessing

# def rotate(image):
#     image = image.reshape([28, 28])
#     image = num.fliplr(image)
#     image = num.rot90(image)
#     return image
#
#
# train_letter = num.asarray(train_letter)
# train_letter = num.apply_along_axis(rotate, 1, train_letter)
#
# test_letter = num.asarray(test_letter)
# test_letter = num.apply_along_axis(rotate, 1, test_letter)
#
#
# test_letter = tf.keras.utils.normalize(test_letter, axis=1)
# train_letter = tf.keras.utils.normalize(train_letter, axis=1)
##############################

# build our model

# model = tf.keras.models.Sequential()

# model.add(tf.keras.layers.Flatten())

# model.add(tf.keras.layers.Dense(units=1024, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))


# model.add(tf.keras.layers.Dense(units=27, activation=tf.nn.softmax))

#######compile our model########

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(train_letter, train_lable, epochs=7)



######predict########

# loss, accuracy = model.evaluate(test_letter, test_lable)
# print(accuracy)
# print(loss)
# model.save('digits.model1')


model1 = tf.keras.models.load_model('digits.model')

letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
          'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
          'w', 'x', 'y', 'z']


#
def prepare(path):
    image = cv.imread(path, cv.IMREAD_GRAYSCALE)
    image2 = cv.resize(image, (28, 28))
    image2 = num.invert(num.array([image2]))
    return image2.reshape(-1, 28, 28, 1)


# image path
pre = model1.predict([prepare("C:\\Users\\FFaye\\OneDrive\\nDesktop\\mid\\AI\\14.jpg")])
print(letter[num.argmax(pre) - 1])
