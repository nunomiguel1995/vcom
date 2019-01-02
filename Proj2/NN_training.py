import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from keras.regularizers import l2
from imutils import paths
import time
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
EPOCHS = 100 # One epoch consists of one full training cycle on the training set.

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
    model.add(Conv2D(64, (5, 5), input_shape=(32, 32, 3), activation="relu") )
    
    model.add(Conv2D(64, (5, 5), activation="relu") )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5), padding="same", activation="relu"))

    model.add(Conv2D(64, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5), padding="same", activation="relu"))

    model.add(Conv2D(64, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(len(label_binarizer.classes_), activation="softmax") )

    return model

def trainingPlot(model):
    # plot the training loss and accuracy
    N = np.arange(0, len(model.history['loss']))
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, model.history["acc"], label="Training accuracy")
    plt.plot(N, model.history["val_acc"], label="Validation accuracy")
    plt.title("Training and Validation Accuracy (CNN)")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig( OUTPUT_PLOT )

def save_fig(fig_id, tight_layout=True):
    path = "nn_output"
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def plot_confusion_matrix(matrix):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    save_fig("confusion_matrix_plot", tight_layout=False)

def getLabel(label_binarizer, predictions):
    # Returns the indices of the maximum values along an axis.
    label_index = predictions.argmax(axis=1)[0]
    label = label_binarizer.classes_[label_index] # Returns the label according to prediction

    return label, label_index

def main():
    startTime = time.time()
    print("[INFO] loading images for training")
    data, labels = loadImages()

    # Split data and labels between 4 arrays
    # 70% for training and 30% for testing
    # scikit-learn -> train_test_split
    (X_train, X_test, y_train, y_test) = train_test_split( data, labels, test_size=0.3, random_state=42 )

    # Keras assumes that labels are encoded as integers and performs
    # one-hot encoding on each label, representing them as vectors
    # rather than integers
    ##### One hot encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction. #####
    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train) # Finds all unique class labels in y_train and transforms them into one-hot encoded labels
    y_test = label_binarizer.transform(y_test) # Performs just the one-hot encoded step (unique set of classes already determined before)

    # Image augmentation allows us to construct “additional” training data from our existing training data 
    # by randomly rotating, shifting, shearing, zooming, and flipping.
    train_datagen = ImageDataGenerator(
        rotation_range=45, 
        width_shift_range=0.2, 
        height_shift_range=0.2, 
        shear_range=0.2, 
        zoom_range=0.2, 
        horizontal_flip=True,
        fill_mode="nearest")

    val_datagen = ImageDataGenerator(
        rotation_range=45, 
        width_shift_range=0.2, 
        height_shift_range=0.2, 
        shear_range=0.2, 
        zoom_range=0.2, 
        horizontal_flip=True,
        fill_mode="nearest")

    train_datagen.fit(X_train, augment=True, seed=42)
    val_datagen.fit(X_test, augment=True, seed=42)

    print("[INFO] defining model")
    # Model definition
    model = modelDefinition(label_binarizer) # Try to make it better

    # Decay -> Learning rate decay over each update
    # Momentum -> Parameter that accelerates SGD in the relevant direction and dampens oscillations
    # nesterov -> Whether to apply nesterov momentum
    sgd = SGD( lr=MODEL_LEARNING_RATE, momentum=0.9, nesterov=False )
    model.compile( loss="categorical_crossentropy", optimizer=sgd , metrics=["accuracy"])

    callback = EarlyStopping(monitor="val_loss", patience=10, verbose=1)

    print("[INFO] training neural network")
    #fit_model = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS )
    fit_model = model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=32), validation_data=val_datagen.flow(X_test, y_test, batch_size=32), validation_steps=len(X_test)/32, steps_per_epoch=len(X_train) / 32, epochs=EPOCHS, callbacks=[callback])

    # Model evaluation
    print("[INFO] evaluating network")
    predictions = model.predict( X_test )
    print( classification_report(y_test.argmax( axis=1 ), predictions.argmax( axis=1), target_names=label_binarizer.classes_ ))

    # Plot the training
    trainingPlot( fit_model )
    
    # Save model
    print("[INFO] serializing model")
    model.save( OUTPUT_MODEL )

    y_test_confusion = [np.argmax(t) for t in y_test]
    y_pred_confusion = [np.argmax(t) for t in predictions]

    confusion = confusion_matrix(y_test_confusion, y_pred_confusion)
    plot_confusion_matrix(confusion)

    file = open( OUTPUT_BIN, "wb" )
    file.write( pickle.dumps(label_binarizer) )
    file.close()   

    endTime = time.time() - startTime

    print("Process took {:.2f} seconds".format(endTime))

if __name__ == '__main__':
    main()