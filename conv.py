import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATA_PATH = "datasets/data.json"


def load_data(data_path):
    """
    loads datasets from json file

    param:
    data_path(str): path to json file 

    return:
    x(ndarray): inputs
    y(ndarray): outputs/targets
    """

    with open(data_path,"r") as fp:
        data = json.load(fp)

    x = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return x, y


def prep_datasets(test_size, valid_size):
    """
    splits loaded datasets into train, validation and test sets for building model

    param:
    test_size: size of test sets
    valid_size: size of validation sets

    return:
    train, valid and test data sets split evenly
    """

    #load data
    x, y = load_data(DATA_PATH)
    
    #create train/test split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = test_size)
    
    #create train/validation split
    x_train, x_valid, y_train, y_valid = train_test_split(x_train,y_train, test_size = valid_size)

    #3d array (number of time bins, nfcc values, channel)
    x_train = x_train[...,np.newaxis] #4d array > (number of samples, and the 3d array)
    x_valid = x_valid[...,np.newaxis]
    x_test = x_test[...,np.newaxis]

    return x_train, x_valid, x_test, y_train, y_valid, y_test


def build_model(input_shape):
    """
    builds model of CNN

    param:
    input_shape: shape of the model 
    """

    #create model
    model = keras.Sequential()

    #1st conv layer
    model.add(keras.layers.Conv2D(32,(3,3),activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3),strides=(2,2), padding ='same'))
    model.add(keras.layers.BatchNormalization())

    #2nd conv layer
    model.add(keras.layers.Conv2D(32,(3,3),activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3),strides=(2,2), padding ='same'))
    model.add(keras.layers.BatchNormalization())

    #3rd conv layer
    model.add(keras.layers.Conv2D(32,(2,2),activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2, 2),strides=(2,2), padding ='same'))
    model.add(keras.layers.BatchNormalization())

    #flatten the output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation = 'relu'))
    model.add(keras.layers.Dropout(0.3))

    #output layer
    model.add(keras.layers.Dense(1,activation='softmax'))

    return model


# def predict(model, x, y):
#     """
#     predicts output from a given data set against trained set

#     param:
#     model: model of CNN
#     x: x data set
#     y: y data set
#     """

#     x = x[np.newaxis,...]
    
#     prediction = model.predict(x)

#     #extract index with max value
#     pred_index = np.argmax(prediction, axis=1) 
#     print("Expected index: {}, Predicted index: {}".format(y, pred_index))


if __name__ == "__main__":
    #create train, validation and test sets (x: inputs, y: outputs)
    x_train, x_valid, x_test, y_train, y_valid, y_test = prep_datasets(0.25, 0.2)

    #build the CNN net
    input_shape = (x_train.shape[1],x_train.shape[2],x_train.shape[3])
    model = build_model(input_shape)

    #compile network 
    optimizer = keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer=optimizer, loss="sparce_categorical_crossentropy",metrics=['accuracy'])

    #train the CNN
    model.fit(x_train,y_train, validation_data=(x_valid,y_valid),batch_size =32, epochs=30)

    #evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    # #make prediction on a sample
    # x = x_test[100]
    # y = y_test[100]

    # predict(model, x, y)




















