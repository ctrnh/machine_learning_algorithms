import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pandas as pd
import os
from sklearn.model_selection import train_test_split
class Digits:
    def __init__(self,):
        folder_path = "./digit-recognizer"
        self.path = os.path.join(folder_path, "train.csv")


    def generate_dataset(self, digits=None):
        all_data_df = pd.read_csv(self.path)
        if digits:
            digits_data_df = all_data_df.loc[all_data_df["label"].isin(digits)]
            self.X = np.array(digits_data_df.drop(columns="label"))
            self.y = np.array(digits_data_df["label"])
        else:
            self.X = np.array(all_data_df)
            self.y = np.array(all_data_df["label"])
        self.X, self.y = shuffle(self.X, self.y)
        return self.X, self.y

    def generate_train_test(self, test_size=0.3, digits=None):
        self.X, self.y = self.generate_dataset(digits=digits)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=test_size)
        return self.X_train, self.X_test, self.y_train, self.y_test


    def display_digits(self, n=5, idx=None):
        if not idx:
            idx = np.random.choice(len(self.X), n)
        assert n <= len(idx)
        for k in range(n):
            i = idx[k]
            plt.subplot(1,n,k+1)
            plt.imshow(np.reshape(self.X[i,:], (28,28)),cmap='gray')
            plt.title(self.y[i])



class DummyDataset2D:
    def __init__(self,
                 means=None,
                 cov=None,
                 N=200):
        self.N = N
        if not means:
            means = [[1,1], [-2,0]]
        if not cov:
            cov = [np.eye(2), np.eye(2)]
        self.means = means
        self.cov = cov

    def generate(self):
        x_pos = np.random.multivariate_normal(mean=self.means[0], cov=self.cov[0], size=(self.N//2))
        x_neg = np.random.multivariate_normal(mean=self.means[1], cov=self.cov[1], size=(self.N//2))
        X = np.vstack((x_pos, x_neg))
        y = [1 for i in range(x_pos.shape[0])] + [0 for i in range(x_neg.shape[0])]
        y = np.array(y)
        self.X, self.y = shuffle(X,y)
        return self.X, self.y

    def plot(self):
        plt.plot(self.X[self.y == 1,0], self.X[self.y == 1,1], 'r*', label="positive")
        plt.plot(self.X[self.y == 0,0], self.X[self.y == 0,1], 'b*', label="negative")
        plt.legend()

    def add(self, x, y):
        self.X = np.vstack((self.X, x))
        self.y = np.hstack((self.y, y))

    def add_rnd_normal(self, mean, cov=None, positive=1, n=5):
        if not cov:
            cov = np.eye(2)
        outliers = np.random.multivariate_normal(mean=mean,
                                                    cov=cov, size=(n))
        self.add(x=outliers, y=[positive for i in range(n)])
