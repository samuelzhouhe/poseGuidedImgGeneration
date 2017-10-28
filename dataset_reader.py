import numpy as np
import nibabel as nib
import FCNConfig as config
from FCN import MRI_HEIGHT, MRI_LENGTH, MRI_WIDTH

class Reader:
    files = []
    images = []
    annotations = []
    batch_offset = 0
    epochs_completed = 0
    subjectIndices = []

    def __init__(self, subjectIndices):
        """
        indices: self-explanatory
        """
        print("Initializing Medical Dataset Reader...")
        self.subjectIndices = subjectIndices  # specific to iseg2017 data.
        self._read_images()

    def _read_images(self):
        self.__channels = True
        self.images = self._getData(self.subjectIndices, 'T1')
        self.__channels = False
        self.annotations = self._getData(self.subjectIndices, 'GT')
        print ("Shape of images (T1 Brain MRI):" , self.images.shape)
        print ("Shape of annotations:", self.annotations.shape)

    def reshapethreedtofourd(self, img):
        return np.reshape(img, [img.shape[0], img.shape[1], img.shape[2]])

    def _getData(self, subjectIndices, type):
        # :param subjectIndices is a list where values are 1to10
        ret = None
        for i in subjectIndices:
            if type == 'GT':
                # ground truth images need to be coded 0 to 3.
                t1 = nib.load(config.data_dir + "/subject-" + str(i)+ "-label.hdr").get_data()
                t1 = np.asarray(t1).reshape([1, t1.shape[0], t1.shape[1], t1.shape[2]])
                t1 = t1.astype("int32")
                t1[t1==10] = 1
                t1[t1==150] = 2
                t1[t1==250] = 3
                pass

                # t1 = t1.astype("float32")
                # t1 = t1 / 255.0 # GT image is from 0 to 255, uint 8.
            elif type == 'T1':
                t1 = nib.load(
                    config.data_dir + "/subject-" + str(i) + "-T1.hdr").get_data()
                t1 = np.asarray(t1).reshape([1, t1.shape[0], t1.shape[1], t1.shape[2]])

                # looks like the T1 image ranges from 0 to 1000. strange.
                t1 = t1.astype("float32")
                t1 = t1 / 1000.0
            # t1 = np.asarray(t1).reshape([1, t1.shape[0], t1.shape[1], t1.shape[2]])
            if ret is None:
                ret = t1
            else:
                ret = np.concatenate((ret, t1), axis=0)
        return ret



    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        # return self.images[start:end], self.annotations[start:end]
        return self.images[start:end, 0:72, 0:96, 0:128], self.annotations[start:end, 0:72, 0:96, 0:128]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]
