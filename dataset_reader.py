import numpy as np
import os
from os import listdir
from os.path import isfile, join
import cv2
import scipy
import itertools
import pickle
import matplotlib.pyplot as plt

IMG_ROOTDIR = "./dataset/Img/img"
KEYPOINTS_ROOTDIR = "./dataset/Img/img-keypoints"

class DataLoader:
    images = None  # (?,256,256,3)
    heatmaps = None  # (?,256,256,18) (also known as "poses")
    morphologicals = None  # (?,256,256)
    pairs = []
    groupsofIndices = []

    def __init__(self):
        print("Initializing DeepFashion Dataset Loader...")
        self._getData()

    def _getData(self):
        for root, dirs, files in os.walk(IMG_ROOTDIR, topdown=True):
            # same directory
            code2index = {}  # code is 01/02/03 etc. Index is 0 through 50000
            for file in files:
                fulldir = root + '/' + file
                if not "flat" in file:
                    img = cv2.imread(fulldir)
                    if img is not None:
                        # perform left-right flip
                        img = np.expand_dims(img, axis=0)
                        flippedImg = np.flip(img, axis=2)
                        # process the keypoint thing
                        heatmap = np.zeros([256, 256, 18])  # (of original image)
                        mapofAllPoints = np.zeros([256, 256])

                        # process the stored keypoints
                        keypointfileDir = fulldir[:fulldir.find('img')+3] + '-keypoints' + fulldir[fulldir.find('img')+3:] + 'keypoints'
                        with open(keypointfileDir, 'rb') as kpfile:
                            keypoints = pickle.load(kpfile)

                            availablePoints = []
                            for i in range(len(keypoints)):
                                keypoint = keypoints[i]

                                # draw circles
                                if len(keypoint) != 0:  # a non-empty keypoint is a
                                    # list consists of one and only one tuple.
                                    availablePoints.append(i)
                                    heatmap[:, :, i] = cv2.circle(np.zeros([256, 256]),
                                                                  (keypoint[0][0], keypoint[0][1]), 4, 255, -1)
                                    cv2.circle(mapofAllPoints, (keypoint[0][0], keypoint[0][1]), 4, 255, -1)

                                    # link the lines
                            links = [(16, 14), (14, 15), (15, 17), (16, 1), (14, 0),
                                     (15, 0), (17, 1), (0, 1), (1, 2),
                                     (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (2, 8), (1, 8), (1, 11), (5, 11),
                                     (8, 9), (9, 10), (8, 11), (11, 12), (12, 13)]
                            for link in links:
                                if link[0] in availablePoints and link[1] in availablePoints:
                                    point1 = (keypoints[link[0]][0][0], keypoints[link[0]][0][1])
                                    point2 = (keypoints[link[1]][0][0], keypoints[link[1]][0][1])
                                    cv2.line(mapofAllPoints, point1, point2, 255, 10)

                        kernel = np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
                        dilatedMapofAllPoints = cv2.dilate(mapofAllPoints, kernel, iterations=6)
                        # quantize it into 1 and 2 (0 becomes 1, 255 becomes 2)
                        dilatedMapofAllPoints[dilatedMapofAllPoints == 0] = 1
                        dilatedMapofAllPoints[dilatedMapofAllPoints == 255] = 2

                        heatmap = np.expand_dims(heatmap, axis=0)
                        heatmap_flippedimg = np.flip(heatmap, axis=2)
                        dilatedMapofAllPoints = np.expand_dims(dilatedMapofAllPoints, axis=0)
                        dilatedMapofAllPoints_flipped = np.flip(dilatedMapofAllPoints, axis=2)
                        # add both images and heatmaps to the their respective grand ndarrays
                        if self.images is None:
                            self.images = np.concatenate([img, flippedImg], axis=0)
                            self.heatmaps = np.concatenate([heatmap, heatmap_flippedimg], axis=0)
                            self.morphologicals = np.concatenate([dilatedMapofAllPoints, dilatedMapofAllPoints_flipped],
                                                                 axis=0)
                        else:
                            self.images = np.concatenate([self.images, img, flippedImg], axis=0)
                            self.heatmaps = np.concatenate([self.heatmaps, heatmap, heatmap_flippedimg], axis=0)
                            self.morphologicals = np.concatenate(
                                [self.morphologicals, dilatedMapofAllPoints, dilatedMapofAllPoints_flipped], axis=0)
                        # code means "which variation of this cloth". Only clothes with the same code are deemed a PAIR.
                        code = file[0:2]
                        if code in code2index:
                            code2index[code].append(len(self.images) - 2)
                            code2index[code].append(len(self.images) - 1)
                        else:
                            code2index[code] = [len(self.images) - 2, len(self.images) - 1]

                        numimgssofar = len(self.images)
                        if numimgssofar% 100 == 0:
                            print(numimgssofar, "Images have been loaded")

        for k, v in code2index.items():
            self.groupsofIndices.append(v)

        # Generate pairs
        for group in self.groupsofIndices:
            self.pairs.append(list(itertools.combinations(group, 2)))
        self.pairs = list(itertools.chain.from_iterable(self.pairs))

    def next_batch(self, batch_size):
        num_pairs = len(self.pairs)
        idx = np.arange(0, num_pairs)
        np.random.shuffle(idx)
        idx = idx[: batch_size]
        conditional_image = np.zeros([batch_size, 256, 256, 3])
        target_pose = np.zeros([batch_size, 256, 256, 18])
        target_image = np.zeros([batch_size, 256, 256, 3])
        target_morphologicals = np.zeros([batch_size, 256, 256])
        for i in range(batch_size):
            indexof_condimg = self.pairs[idx[i]][0]
            indexof_targetimg = self.pairs[idx[i]][1]
            conditional_image[i, :, :, :] = self.images[indexof_condimg, :, :, :]
            target_pose[i, :, :, :] = self.heatmaps[indexof_targetimg, :, :, :]
            target_image[i, :, :, :] = self.images[indexof_targetimg, :, :, :]
            target_morphologicals[i, :, :] = self.morphologicals[indexof_targetimg, :, :]
        g1_feed = np.concatenate([conditional_image, target_pose], axis=3)  # the (batch,256,256,21) thing.
        return g1_feed, conditional_image,target_image, np.expand_dims(target_morphologicals,axis=3)
