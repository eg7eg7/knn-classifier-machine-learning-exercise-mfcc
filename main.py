# In this exercise you will implement your first word recognizer using the KNN
# algorithm. Your algorithm will recognize the digits 1 - 5. For that, You will
# implement the KNN algorithm, where the distance metric should be the L2.
#
# EDEN DUPONT 204808596
import os

import librosa
import numpy as np
import glob
from scipy.stats import stats
import sksound
from sksound.sounds import Sound

test_files_path = r"D:/GitHub/knn-exercise/test_files"

training_data_path = r"D:/GitHub/knn-exercise/train_data/"
training_folders = ["one", "two", "three", "four", "five"]
output_file = "output.txt"


class KnnClassifier:
    INFINITE_DISTANCE = 1000000.0

    def __init__(self, dim_size, k=3):
        self.k = 0
        self.dim_size = dim_size
        self.num_classes = 0
        self.k = k
        self.training_dict = {}

    def set_k(self, k):
        if not ((self.num_classes == 2 and (k % 2 == 1)) or (k % self.num_classes != 0)):
            print("WARNING: K must be odd, if knn contains 2 classes, and it must not be a multiple of the number of classes")
        self.k = k

    def predict_class(self, mfcc_input):
        closest_distances = [('label', KnnClassifier.INFINITE_DISTANCE) for i in range(self.k)]
        for label, mfcc_class in self.training_dict.items():
            for mfcc in mfcc_class:
                distance = np.linalg.norm(mfcc - mfcc_input)  # L2 Distance ( Euclidean )
                for old_label, dist in closest_distances:
                    if distance < dist:
                        closest_distances.append((label, distance))
                        break
                closest_distances.sort(key=lambda tup: tup[1])
                closest_distances = closest_distances[0:self.k]

        predict_dict = {}
        for label, dist in closest_distances:
            if label not in predict_dict.keys():
                predict_dict[label] = 0
            predict_dict[label] += 1
        return max(predict_dict, key=lambda key: predict_dict[key])

    def add_training_data_point(self, mfcc, label):
        if label not in self.training_dict.keys():
            self.training_dict[label] = []
            self.num_classes += 1
            self.set_k(self.k)  # checking for warnings
        self.training_dict[label].append(mfcc)

    def print_training_data(self, verbose=False):
        for key in self.training_dict.keys():
            print(key)
            if verbose:
                print(self.training_dict[key])


def extract_mfcc_from_wav(wav_file_path):
    y, sr = librosa.load(wav_file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc = stats.zscore(mfcc, axis=1)  # Normalize
    return mfcc


def add_training_dataset(knn):
    for dataset in training_folders:
        path = r'%s' % (training_data_path + dataset)
        for filename in os.listdir(path):
            if filename.endswith(".wav"):
                mfcc = extract_mfcc_from_wav(path + "/" + filename)
                knn.add_training_data_point(mfcc=mfcc, label=dataset)



def playsound(path):
    mySound = Sound(path)
    mySound.play()


def predict_classes(knn):
    for file in os.listdir(test_files_path):
        if file.endswith(".wav"):
            path = test_files_path + "/" + file
            mfcc = extract_mfcc_from_wav(path)
            a = knn.predict_class(mfcc)
            print(a)
            # playsound(path)


def main():
    knn = KnnClassifier(dim_size=20, k=3)
    add_training_dataset(knn)
    # print(knn.training_dict['one'][0])
    predict_classes(knn)


if __name__ == "__main__":
    main()
