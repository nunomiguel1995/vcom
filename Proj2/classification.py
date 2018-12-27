import numpy as np
from os.path import isfile, join
from os import listdir
from random import shuffle
from PIL import Image

CLASSES = ['arrabida', 'camara', 'clerigos', 'musica', 'serralves']

class DataSetGenerator:
    def __init__(self, directory): # Directory should be Dataset/images
        self.directory = directory
        self.dictionary = {}
        self.labels = self.getLabels()
        self.paths = self.getPaths()

    def getLabels(self):
        labels = []
        for filename in listdir(self.directory):
            if not isfile(join(self.directory, filename)):
                labels.append(filename)
        return labels

    def getPaths(self):
        paths = []
        for label in self.labels:
            img_lists = []
            path = join(self.directory, label)
            for filename in listdir(path):
                image_path = join(path, filename)
                img_lists.append(image_path)

            paths.append(img_lists)

        paths = [item for sublist in paths for item in sublist] # Flatten the paths array

        return paths

def openImage(path):
    im = Image.open(path)
    im = (np.array(im))

    r = im[:, :, 0].flatten()
    g = im[:, :, 1].flatten()
    b = im[:, :, 2].flatten()
    label = [1]

    out = np.array(list(label) + list(r) + list(g) + list(b), np.uint8)

    return out

def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def main():
    generator = DataSetGenerator("Dataset/images")

    data = []
    labels = []
    dictionary = {}

    for label in generator.labels:
        labels.append(label)

    for path in generator.paths:
        output = openImage(path)
        data.append(output)

    dictionary["data"] = data
    dictionary["labels"] = labels

    save_obj(dictionary, 'dataset')

    print(generator.labels)

if __name__ == '__main__':
    main()