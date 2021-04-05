import numpy as np
import matplotlib.pyplot as plt
class BinaryClassifier:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.X_train = None
        self.y_train = None

    def initialize(self, X_train, y_train, add_intercept=False):
        self.y_train = y_train
        self.n = self.y_train.shape[0]
        if add_intercept:
            self.X_train = np.hstack((X_train, np.ones((self.n,1))))
        else:
            self.X_train = X_train
        self.d = self.X_train.shape[1]
        assert self.X_train.shape[0] == self.n



    def predict_class(self, X_test):
        raise NotImplementedError

    def evaluate(self, X_test, y_test):
        predictions = self.predict_class(X_test)
        accuracy =  np.sum(predictions == y_test)/len(y_test)

        tp = np.sum((predictions == y_test) & (y_test == 1))
        fp = np.sum((predictions == 1) & (y_test == 0))
        fn = np.sum((predictions == 0) & (y_test == 1))

        precision = tp / (fp + tp)
        recall = tp / (fn + tp)
        return accuracy, precision, recall


    def plot_2D(self, X_test, y_test):
        x_pos = X_test[y_test == 1, :]
        x_neg = X_test[y_test == 0, :]

        pred = self.predict_class(X_test)

        plt.plot(x_pos[:,0], x_pos[:,1], 'o',color='tomato', label="true positive")
        plt.plot(x_neg[:,0], x_neg[:,1], 'o', color="skyblue", label="true negative")

        plt.plot(X_test[pred == 1,0],X_test[pred == 1,1], 'x', color="firebrick", label="predicted positive" )
        plt.plot(X_test[pred == 0,0],X_test[pred == 0,1], 'x', color="blue" , label="predicted negative")

        self.visualize(X_test=X_test)

        plt.legend()
