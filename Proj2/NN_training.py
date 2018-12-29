import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import cv2 as cv
import os

DATASET_IMAGES = "Dataset/images"
OUTPUT_MODEL = "nn_output/data_model.model"
OUTPUT_BIN = "nn_output/label_binarizer.pickle"
OUTPUT_PLOT = "nn_output/plot.png"

MODEL_LEARNING_RATE = 0.01
EPOCHS = 5 # One epoch consists of one full training cycle on the training set.

def loadImages():
    data = []
    labels = []

    images = sorted(list(paths.list_images(DATASET_IMAGES)))
    random.seed(42)
    random.shuffle(images)

    counter = 1

    for path in images:
        print("[LOADING] Loading image number " + str(counter))
        counter = counter + 1

        image = cv.imread(path)
        image = cv.resize(image, (32, 32))
        data.append(image)
 
        label = path.split(os.path.sep)[-2]
        labels.append(label)

    data = np.array(data, dtype="float") / 255.0 # raw pixel from [0, 255] to [0, 1]
    labels = np.array(labels)

    return data, labels

def modelDefinition(label_binarizer):
    model = Sequential()

    # Convolutional Neural Network
    model.add( Conv2D(32, (3, 3), input_shape=(32,32,3)) )
    model.add( Activation("relu") )

    model.add( Conv2D(32, (3, 3)) )
    model.add( Activation("relu") )

    model.add( Conv2D(64, (3, 3)) )
    model.add( Activation("relu") )

    model.add( Flatten() )
    model.add( Dense(64) )
    model.add( Activation('relu') )
    model.add( Dropout(0.5) )
    model.add( Dense(len(label_binarizer.classes_)) )
    model.add( Activation('sigmoid') )

    return model

def trainingPlot(model):
    # plot the training loss and accuracy
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, model.history["loss"], label="train_err")
    plt.plot(N, model.history["val_loss"], label="val_err")
    plt.plot(N, model.history["acc"], label="train_acc")
    plt.plot(N, model.history["val_acc"], label="val_acc")
    plt.title("Training Error and Accuracy (Simple NN)")
    plt.xlabel("Epoch #")
    plt.ylabel("Error/Accuracy")
    plt.legend()
    plt.savefig( OUTPUT_PLOT )

def main():
    print("[INFO] loading images for training")
    data, labels = loadImages()

    # Split data and labels between 4 arrays
    # 100% for training and 0% for testing
    # scikit-learn -> train_test_split
    (X_train, X_test, y_train, y_test) = train_test_split( data, labels, test_size=0.15, random_state=42 )

    # Keras assumes that labels are encoded as integers and performs
    # one-hot encoding on each label, representing them as vectors
    # rather than integers
    ##### One hot encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction. #####
    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train) # Finds all unique class labels in y_train and transforms them into one-hot encoded labels
    y_test = label_binarizer.transform(y_test) # Performs just the one-hot encoded step (unique set of classes already determined before)

    print("[INFO] defining model")
    # Model definition
    model = modelDefinition(label_binarizer) # Try to make it better

    # Decay -> Learning rate decay over each update
    # Momentum -> Parameter that accelerates SGD in the relevant direction and dampens oscillations
    # nesterov -> Whether to apply nesterov momentum
    sgd = SGD( lr=MODEL_LEARNING_RATE, decay=1e-6, momentum=0.9, nesterov=True )
    model.compile( loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])

    print("[INFO] training neural network")
    fit_model = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS )

    # Model evaluation
    print("[INFO] evaluating network")
    predictions = model.predict( X_test )
    print( classification_report(y_test.argmax( axis=1 ), predictions.argmax( axis=1), target_names=label_binarizer.classes_ ))

    # Plot the training
    trainingPlot( fit_model )
    
    # Save model
    print("[INFO] serializing model")
    model.save( OUTPUT_MODEL )

    file = open( OUTPUT_BIN, "wb" )
    file.write( pickle.dumps(label_binarizer) )
    file.close()   

if __name__ == '__main__':
    main()