import numpy as np
import os
from os import listdir
from os.path import isfile, join
import cv2
import scipy
import itertools

ROOTDIR = "../clothes/Img/img/WOMEN/Denim"

class DataLoader:
    images = None
    pairs=[]
    groupsofIndices = []


    def __init__(self):
        print("Initializing DeepFashion Dataset Loader...")
        self._read_images()

    def _read_images(self):
        self.images = self._getData()

    def _getData(self):
        for root, dirs, files in os.walk(ROOTDIR, topdown=True):
            # same directory
            code2index = {} # code is 01/02/03 etc. Index is 0 through 50000
            for file in files:
                fulldir = root + '/' + file
                if not "flat" in file:
                    img = cv2.imread(fulldir)
                    if img is not None:
                        img = np.expand_dims(img,axis=0)
                        flippedImg = np.flip(img, axis=2)
                        if self.images is None:
                            self.images = np.concatenate([img, flippedImg],axis=0)
                            pass
                        else:
                            self.images = np.concatenate([self.images,img,flippedImg],axis=0)
                        code = file[0:2]
                        if code in code2index:
                            code2index[code].append(len(self.images)-2)
                            code2index[code].append(len(self.images)-1)
                        else:
                            code2index[code] = [len(self.images)-2,len(self.images)-1]
                        pass
            for k,v in code2index.items():
                self.groupsofIndices.append(v)
        for group in self.groupsofIndices:
            self.pairs.append(list(itertools.combinations(group,2)))
        self.pairs = list(itertools.chain.from_iterable(self.pairs))
        print(len(self.pairs))
        pass

    def next_batch(self, batch_size):
        pass

loader = DataLoader()