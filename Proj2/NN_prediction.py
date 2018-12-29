import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2 as cv
import os

INPUT_MODEL = "nn_output/data_model.model"
INPUT_BIN = "nn_output/label_binarizer.pickle"

def getImagePath():
    # Must pass image path by argument, therefore, must parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to input image to classify")

    args = vars( ap.parse_args() )
    imagePath = args["image"]

    return imagePath

def processImage(image):
    output = image.copy()
    image = cv.resize(image, (32, 32))

    image = image.astype("float") / 255.0 # raw pixel from [0, 255] to [0, 1]

    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) # 1-D array with dimension 4

    return image

def getLabel(label_binarizer, predictions):
    # Returns the indices of the maximum values along an axis.
    label_index = predictions.argmax(axis=1)[0]
    label = label_binarizer.classes_[label_index] # Returns the label according to prediction

    return label, label_index

def main():
    imagePath = getImagePath()

    originalImage = cv.imread(imagePath)
    processedImage = processImage(originalImage)

    print("[INFO] loading model")
    # Loads model and the binarized labels
    model = load_model(INPUT_MODEL)
    label_binarizer = pickle.loads(open(INPUT_BIN, "rb" ).read())

    # Predicts what the processed image is
    predictions = model.predict(processedImage)

    label, label_index = getLabel(label_binarizer, predictions)

    # Formats the label (label: percentage) and puts it on the original image
    text = "{}: {:.2f}%".format(label, predictions[0][label_index] * 100)
    cv.putText(originalImage, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv.imshow(label, originalImage)
    cv.waitKey(0)

if __name__ == '__main__':
    main()