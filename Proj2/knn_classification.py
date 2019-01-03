import os
import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import time
import glob

# Opens an image, converts it to gray scale and changes its size to 32 x 32 pixels
def openImage(filename):
    image = cv.imread(filename)
    try:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    except:
        print(" --(!) Error reading image ", filename)
        return None
    return cv.resize(image, (32, 32))

# Loads and processes all images of a category
def loadFileNames(name):
    rootdir = "porto-dataset/images/" + name + "/*.jpg"
    files = glob.glob(rootdir)
    labels = []
    data = []
    for count in range (0, len(files)):
        if (count % 10 == 0):
            print("Loading image {}/{} of {} folder.".format(count, len(files), name))
        labels.append(name)
        img = openImage(files[count])
        data.append(img.flatten())
    return data, labels

def save_fig(fig_id, tight_layout=True):
    path = "images"
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

  # Creates the confusion matrix and saves it in a file  
def plot_confusion_matrix(matrix):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    save_fig("confusion_matrix_plot", tight_layout=False)

# Using Grid and Randomized Search, tries to find the hyperparameters that give the best accuracy for the model
def findBestParams(model, X_train, y_train, X_test, y_test):
    params = {"n_neighbors": np.arrange(1,25,2),
	"metric": ["cityblock", "euclidean"]}

    grid1 = GridSearchCV(model, params)

    start = time.time()
    grid1.fit(X_train, y_train)

    acc1 = grid1.score(X_test, y_test)
    print("Grid Search accuracy = {}".format(acc1))
    print("Grid Search best params = {}".format(grid1.best_params_))
    print("Grid Search took {:.2f} seconds".format(time.time() - start))

    grid2 = RandomizedSearchCV(model_knn, params)

    start = time.time()
    grid2.fit(X_train, y_train)

    acc2 = grid2.score(X_test, y_test)
    print("Randomized Search accuracy = {}".format(acc2))
    print("Randomized Search best params = {}".format(grid2.best_params_))
    print("Randomized Search took {:.2f} seconds".format(time.time() - start))

def loadData():
    ############### LOADING THE DATA #################
    X = []
    y = []
    print("Loading all the data")
    paths = ['arrabida', 'musica', 'clerigos', 'camara', 'serralves']
    for count in range (0, len(paths)):
        data, labels = loadFileNames(paths[count])
        X.extend(data)
        y.extend(labels)

    X = np.array(X)
    y = np.array(y)

    (X_train, X_test, y_train, y_test) = train_test_split(
	X, y, test_size=0.3, random_state=42)
    print("Finished loading the data")
    return X_train, X_test, y_train, y_test

# Applies the KNN algorithm to the data loaded
def knnModel():
    ############### TEST WITH KNN CLASSIFIER #################
    X_train, X_test, y_train, y_test = loadData()

    print('calculating classifier...')

    model_knn = KNeighborsClassifier(n_neighbors=1, metric="cityblock")
    model_knn.fit(X_train, y_train)

    acc1 = model_knn.score(X_test, y_test)
    print("Accuracy = {}".format(acc1))

    y_train_pred = cross_val_predict(model_knn, X_train, y_train, cv=3)

    cm = confusion_matrix(y_train, y_train_pred)
    plot_confusion_matrix(cm)

    return model_knn

def makePrediction(model):
    x = "1"

    while (x != "0"):
        x = raw_input("Type the name of the image to make a prediction (Type 0 to quit): ")
        if (x != "0"):
            image = openImage(x).flatten()
            prediction = model.predict([image])
            print("The prediction was {}.".format(prediction[0]))

def main():
    model = knnModel()
    makePrediction(model)

if __name__ == '__main__':
    main()