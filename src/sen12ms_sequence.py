import os
import math
import numpy as np
import tensorflow as tf
import progressbar
from multiprocessing.pool import ThreadPool

from sen12ms_dataLoader import SEN12MSDataset, Seasons, S1Bands, S2Bands, LCBands

sen12ms_path = os.path.join(os.path.dirname(__file__), "..", "data")

seasons = [Seasons.SPRING, Seasons.SUMMER, Seasons.FALL, Seasons.WINTER]
classes = {
    1   : [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # Evergreen Needleleaf Forests
    2   : [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # Evergreen Broadleaf Forests
    3   : [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # Deciduous Needleleaf Forests
    4   : [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],  # Deciduous Broadleaf Forests
    5   : [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],  # Mixed Forests
    6   : [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],  # Closed (Dense) Shrubland
    7   : [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],  # Open (Sparse) Shrubland
    8   : [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],  # Woody Savannas
    9   : [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],  # Savannas
    10  : [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],  # Grasslands
    11  : [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],  # Permanent Wetlands
    12  : [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],  # Croplands
    13  : [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],  # Urban and Built-Up Lands
    14  : [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],  # Croplands/Natural Vegetation Mosaics
    15  : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],  # Permanent Snow and Ice
    16  : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],  # Barren
    17  : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]   # Water Bodies
}

def flatten_image(image : np.array):
    out = np.zeros((image.shape[0],image.shape[1]))
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[0]):
            out[x][y] = image[x][y][0]
    return out

def normalize_image(image : np.array):
    image = image.astype(float) / float((2.0**16.0)-1.0)
    return image

def reorder_image(image : np.array):
    image = np.swapaxes(image, 0, 1)
    image = np.swapaxes(image, 1, 2)
    return image

def onehot_decode(image : np.array):

    y_decode = np.zeros(shape=(image.shape[0], image.shape[1]))

    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[0]):
            
            for key in classes:
                if np.all(np.equal(image[x][y].astype(int),np.array(classes[key]).astype(int))):
                    y_decode[x][y] = key

    return y_decode

def onehot_encode(y : np.array):

    y_onehot = np.zeros(shape=(y.shape[0], y.shape[1], y.shape[2], len(classes)))

    for i in range(0, y.shape[0]):
        for j in range(0,y.shape[1]):
            for k in range(0,y.shape[2]):
                if y[i][j][k][0] == 0:
                    y_onehot[i][j][k] = classes[10]
                else:
                    y_onehot[i][j][k] = classes[y[i][j][k][0]]

    return y_onehot

def _thread_calculate_class_weights(args):
    class_weights = {}
    for key in classes.keys():
        class_weights[key-1] = 0

    sen12ms : SEN12MSDataset = args[0]
    data = args[1]

    # Load triplet from index
    lc = sen12ms.get_patch(
        seasons[int(data[0])],
        int(data[1]),
        int(data[2]),
        LCBands.IGBP)

    # Get the counts of each class
    unique, counts = np.unique(lc[0], return_counts=True)
            
    # Update the class weights with the counts
    for i in range(0,len(unique)):
        if unique[i]-1 in class_weights.keys():
            class_weights[unique[i]-1] += counts[i]
        else:
            class_weights[unique[i]-1] = counts[i]

    # Increase total samples by the number of pixels
    samples = 256 * 256

    return class_weights, samples

class SEN12MSSequence(tf.keras.utils.Sequence):

    def __init__(self, data=None, batch_size=32):
        self.sen12ms = SEN12MSDataset(sen12ms_path)
        self.data = data
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(self.data.shape[0] / self.batch_size)
    
    def __getitem__(self, idx):
        x : np.array = np.zeros(shape=(self.batch_size, 256, 256, 15))
        y : np.array = np.zeros(shape=(self.batch_size, 256, 256, 1))

        start_idx = self.batch_size * idx
        end_idx = min(start_idx + self.batch_size, self.data.shape[0])
        
        for dataset_idx in range(start_idx, end_idx):
            
            # Load triplet from index
            s1, s2, lc, bounds = self.sen12ms.get_s1s2lc_triplet(
                seasons[int(self.data[dataset_idx][0])], 
                int(self.data[dataset_idx][1]), 
                int(self.data[dataset_idx][2]), 
                s1_bands=S1Bands.ALL,
                s2_bands=S2Bands.ALL, 
                lc_bands=LCBands.IGBP)

            s1 = reorder_image(s1)
            s2 = reorder_image(s2)
            lc = reorder_image(lc)

            row_x = np.dstack([s2, s1])
            row_y = lc

            x[dataset_idx - start_idx] = row_x
            y[dataset_idx - start_idx] = row_y

        # Do any preprocessing steps on the batches

        x = normalize_image(x)
        y = onehot_encode(y)

        return x, y


    def calculate_class_weights(self):
        self.class_weights = {}
        
        # Get all the patches, count occurences of pixel categories
        total_samples = 0
        bar = progressbar.ProgressBar(0, self.data.shape[0], widgets=['[',progressbar.SimpleProgress(),']',progressbar.Percentage()])
        
        # Create thread arg list
        args = []
        for i in range(0, self.data.shape[0]):
            args.append((self.sen12ms, self.data[i][:]))

        # Multithreaded label counting
        with ThreadPool(128) as pool:
            for result in pool.imap(_thread_calculate_class_weights, args):
                # Merge results
                class_weights = result[0]
                samples = result[1]

                for key in class_weights:
                    if key in self.class_weights.keys():
                        self.class_weights[key] += class_weights[key]
                    else:
                        self.class_weights[key] = 0

                total_samples += samples
                
                bar.update(bar.currval + 1)

        # Convert counts into class weights
        for key in self.class_weights.keys():
            self.class_weights[key] = float(total_samples) / (float(len(self.class_weights.keys())) * float(self.class_weights[key]))

    def get_class_weights(self):
        return self.class_weights
            
    def get_item(self, dataset_idx):
        
        s1, s2, lc, bounds = self.sen12ms.get_s1s2lc_triplet(
            seasons[int(self.data[dataset_idx][0])], 
            int(self.data[dataset_idx][1]), 
            int(self.data[dataset_idx][2]), 
            s1_bands=S1Bands.ALL,
            s2_bands=S2Bands.ALL, 
            lc_bands=LCBands.IGBP)

        s1 = reorder_image(s1)
        s2 = reorder_image(s2)
        lc = reorder_image(lc)

        x = normalize_image(np.dstack([s2, s1]))
        y = flatten_image(lc)

        return x, y