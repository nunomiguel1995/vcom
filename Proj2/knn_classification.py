import os
import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import time

def openImage(filename):
    image = cv.imread(filename)
    try:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    except:
        print(" --(!) Error reading image ", filename)
        return None
    return cv.resize(image, (64, 64))

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
    
def plot_confusion_matrix(matrix):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    save_fig("confusion_matrix_plot", tight_layout=False)

def loadData():
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

    print("Finished loading the data")

    ############### TEST WITH SGD CLASSIFIER #################
    (X_train, X_test, y_train, y_test) = train_test_split(
	X, y, test_size=0.3, random_state=42)

    print('calculating classifier...')

    model_knn = KNeighborsClassifier()
    params = {"n_neighbors": [1],
	"metric": ["cityblock"]}

    grid1 = GridSearchCV(model_knn, params)

    start = time.time()
    grid1.fit(X_train, y_train)

    acc1 = grid1.score(X_test, y_test)
    print("Grid Search accuracy = {}".format(acc1))
    print("Grid Search best params = {}".format(grid1.best_params_))
    print("Grid Search took {:.2f} seconds".format(time.time() - start))

    #grid2 = RandomizedSearchCV(model_knn, params)

    #start = time.time()
    #grid2.fit(X_train, y_train)

    #acc2 = grid2.score(X_test, y_test)
    #print("Randomized Search accuracy = {}".format(acc2))
    #print("Randomized Search best params = {}".format(grid2.best_params_))
    #print("Randomized Search took {:.2f} seconds".format(time.time() - start))
    #print('applied classifier, calculating prediction...')

    #some_digit = openImage('test10.PNG').flatten()

    #pred = grid1.predict([some_digit])
    #print("Prediction: {}".format(pred[0]))

    y_train_pred = cross_val_predict(grid1, X_train, y_train, cv=3)

    cm = confusion_matrix(y_train, y_train_pred)
    plot_confusion_matrix(cm)


    
def main():
    loadData()

if __name__ == '__main__':
    main()