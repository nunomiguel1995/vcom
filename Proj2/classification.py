import numpy as np
from os.path import isfile, join
from os import listdir
from random import shuffle

CLASSES = ['arrabida', 'camara', 'clerigos', 'musica', 'serralves']

class DataSetGenerator:
    # Directory should be Dataset/images
    def __init__(self, directory):
        self.directory = directory
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
                tokens = filename.split('.')
                if tokens[-1] == 'jpg' or tokens[-1] == 'jpeg' or tokens[-1] == 'JPG':
                    image_path = join(path, filename)
                    img_lists.append(image_path)
            paths.append(img_lists)

        paths = [item for sublist in paths for item in sublist] # Flatten the paths array

        shuffle(paths)
        print(len(paths))
        return paths

def main():
    generator = DataSetGenerator("Dataset/images")

if __name__ == '__main__':
    main()