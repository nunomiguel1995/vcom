import xml.etree.ElementTree as ET
import csv
import os

DATASET_ANNOTATIONS = "Dataset/annotations/"

# Returns all the files in the dataset folder and subfolder
def get_file_list():
    lf = list()
    for path, subdirs, files in os.walk(DATASET_ANNOTATIONS):
        lf += [os.path.join(path, file) for file in files]
    
    return lf

# Processes the annotations to a special format and saves on a text file
# file_path xmin,ymin,xmax,ymax,class_id
def read_annotations():
    files = get_file_list()
    with open('annotations.txt', 'w', newline='') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_NONE, delimiter=" ")

        # Iterates over every xml_file
        for xml_file in files:
            print("[LOADING] Parsing file", xml_file)
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for element in root.iter('annotation'):
                obj = element.find('object')
                bndbox = obj.find('bndbox')

                filename = element.find('filename').text
                xmin = bndbox.find('xmin').text
                ymin = bndbox.find('ymin').text
                xmax = bndbox.find('xmax').text
                ymax = bndbox.find('ymax').text
                box = xmin + "," + ymin + "," + xmax + "," + ymax
                label = obj.find('name').text
                path = "../Dataset/images/" + label + "/" + filename

                if label == 'arrabida':
                    label = 0
                elif label == 'camara':
                    label = 1
                elif label == 'clerigos':
                    label = 2
                elif label == 'musica':
                    label = 3
                elif label == 'serralves':
                    label = 4

                box = box + "," + str(label)
                line = (path,box)

                wr.writerow(line)

def main():
    read_annotations()

if __name__ == '__main__':
    main()