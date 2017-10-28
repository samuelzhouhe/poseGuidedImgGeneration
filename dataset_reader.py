import numpy as np


class DataLoader:
    images = []

    def __init__(self, subjectIndices):
        print("Initializing DeepFashion Dataset Loader...")
        self._read_images()

    def _read_images(self):
        self.images = self._getData()

    def _getData(self):
        pass

    def next_batch(self, batch_size):
        pass