from tensorflow import keras
import keras
from keras.models import sequential
from keras.layers import Dense, Activation
import numpy as np
from dataset import load_hoda
from keras.utils import np_utils


np.random.seed(123)

x_train_original, y_train_original, x_test_original, y_test_original = load_hoda()

def print_data_info(x_train, y_train, x_test, y_test):

# check data type

    print ("\ttype (x_train):{}".format(type(x_train)))
    print ("\ttype (y_train):{}".format(type(y_train)))

# check data shape


    print ("\tx_train.shape:{}".format(np.shape(x_train)))
    print ("\ty_train.shape:{}".format(np.shape(y_train)))
    print ("\tx_test.shape:{}".format(np.shape(x_test)))
    print ("\ty_test.shape:{}".format(np.shape(y_test)))

# Sample data

    print ("\ty_train[0]:{}".format(y_train[0]))


x_train = np.array(x_train_original)
y_train = np_utils.to_categorical(y_train_original, num_classes=10)


x_test = np.array((x_test_original))
y_test = np_utils.to_categorical(y_test_original, num_classes=10)

print ("Befor Preprocessing")
print_data_info(x_train_original, y_train_original, x_test_original, y_test_original)

print ("After Preprocessing")
print_data_info(x_train, y_train, x_test, y_test)
