from __future__ import print_function

import sys
import os, sys, tarfile
import numpy as np
import matplotlib.pyplot as plt
    
if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib # ugly but works
else:
    import urllib

print(sys.version_info) 

# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

# current file dir
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# path to the directory with the data
DATA_DIR = DIR_PATH + '/../../images/'

# url of the binary data
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

# paths to the binary train file with image data and label
TR_DATA_PATH = DIR_PATH + '/data/stl10_binary/train_X.bin'
TR_LABEL_PATH = DIR_PATH + '/data/stl10_binary/train_y.bin'

# path to binary unlabeld 
UNLB_DATA_PATH = DIR_PATH + '/../../images/stl10_binary/unlabeled_X.bin'

# paths to test file with image data and label
TS_DATA_PATH = DIR_PATH + '/../../images/stl10_binary/test_X.bin'
TS_LABEL_PATH = DIR_PATH + '/../../images/stl10_binary/test_y.bin'



def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data, ret_count=-1):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))
        if ret_count != -1:
            image = images[:ret_count]

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def read_single_image(image_file):
    """
    CAREFUL! - this method uses a file as input instead of the path - so the
    position of the reader will be remembered outside of context of this method.
    :param image_file: the open file containing the images
    :return: a single image
    """
    # read a single image, count determines the number of uint8's to read
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
    # force into image matrix
    image = np.reshape(image, (3, 96, 96))
    # transpose to standard format
    # You might want to comment this line or reverse the shuffle
    # if you will use a learning algorithm like CNN, since they like
    # their channels separated.
    image = np.transpose(image, (2, 1, 0))
    return image


def plot_image(image):
    """
    :param image: the image to be plotted in a 3-D matrix format
    :return: None
    """
    plt.imshow(image)
    plt.show()


def download_and_extract():
    """
    Download and extract the STL-10 dataset
    :return: None
    """
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print('Downloaded', filename)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def rgb2gray(X):
    r, g, b = X[..., 0], X[...,1], X[...,2]
    X = (0.2125 * r) + (0.7154 * g) + (0.0721 * b)
    X = X[..., np.newaxis]
    return X

def load_test(grayscale=False):
    test_images = read_all_images(TS_DATA_PATH)
    test_labels = read_labels(TS_LABEL_PATH)
    if grayscale:
        test_images = rgb2gray(test_images)
    return test_images, test_labels

def load_data(grayscale=False, unlabel_count=-1):
    """
    unlabel_count: -1 is all any other value is the amount returned
    """
    download_and_extract()
    
    train_images = read_all_images(TR_DATA_PATH)
    train_labels = read_labels(TR_LABEL_PATH)
    unlabel_images = np.array([])
    if unlabel_count != 0:
        unlabel_images = read_all_images(UNLB_DATA_PATH, unlabel_count)
    
    if grayscale:
        train_images = rgb2gray(train_images)
        if unlabel_count != 0:
            unlabel_images = rgb2gray(unlabel_images)
            train_images  = np.concatenate((train_images, unlabel_images), axis=0)
        test_images, test_labels = load_test(grayscale)
        print('test shape {}'.format(test_images.shape))
    return train_images, test_images

if __name__ == "__main__":
    # download data if needed
    download_and_extract()

    # test to check if the image is read correctly
    with open(DATA_PATH) as f:
        image = read_single_image(f)
        plot_image(image)

    # test to check if the whole dataset is read correctly
    images = read_all_images(DATA_PATH)
    print(images.shape)

    labels = read_labels(LABEL_PATH)
    print(labels.shape)
