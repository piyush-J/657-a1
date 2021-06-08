import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split


class MLP_Q4:
    def __init__(self, hidden_nodes: int):  # hidden_nodes - hyperparameter
        self.output_classes = 4  # 4 output classes - as per the problem
        self.layers_size = [hidden_nodes, self.output_classes]
        self.parameters = {}  # to store weights/biases
        self.L = 2
        self.n = 0
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

    @staticmethod
    def sigmoid(Z):  # formula for sigmoid
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def softmax(Z):  # formula for softmax
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def init_model(self):
        np.random.seed(1)  # setting seed value
        for layer_index in range(1, len(self.layers_size)):  # initializing the weights as random and biases as zeros
            self.parameters["W" + str(layer_index)] = np.random.randn(self.layers_size[layer_index],
                                                                      self.layers_size[layer_index - 1])
            self.parameters["b" + str(layer_index)] = np.zeros((self.layers_size[layer_index], 1))

    def forward(self, x):
        parameters_ = {}

        # between input and hidden layer
        a_ = self.parameters["W1"].dot(x.T) + self.parameters["b1"]  # Wx^T+b
        f_ = self.sigmoid(a_)  # sigmoid activation for hidden layer
        parameters_["f_1"] = f_
        parameters_["W1"] = self.parameters["W1"]
        parameters_["a_1"] = a_

        # between hidden and output layer
        a_ = self.parameters["W2"].dot(f_) + self.parameters["b2"]  # second layer
        f_ = self.softmax(a_)  # softmax activation for output layer
        parameters_["f_2"] = f_
        parameters_["W2"] = self.parameters["W2"]
        parameters_["a_2"] = a_

        return f_, parameters_

    def backward(self, X_train, y_train, parameter_):
        derivatives = {}  # to store dE/dW_2,ndE/dW_1, dE/db_2, dE/db_1 for weight updation
        parameter_["f_0"] = X_train.T
        f = parameter_["f_2"]  # y-hat from forward pass
        df = f - y_train.T  # derivative of cross entropy loss with softmax

        dW = df.dot(parameter_["f_1"].T) / self.n  # dE/dW_2
        db = np.sum(df, axis=1, keepdims=True) / self.n  # dE/db_2
        df_prev = parameter_["W2"].T.dot(df)  # will be used by the previous layer

        derivatives["dW2"] = dW
        derivatives["db2"] = db

        # calculate the sigmoid derivative
        s = 1 / (1 + np.exp(-parameter_["a_1"]))
        sigmoid_derv = s * (1 - s)

        df = df_prev * sigmoid_derv
        dW = 1. / self.n * df.dot(parameter_["f_0"].T)  # dE/dW_1
        db = 1. / self.n * np.sum(df, axis=1, keepdims=True)  # dE/db_1

        derivatives["dW1"] = dW
        derivatives["db1"] = db

        return derivatives

    def fit(self, X_train, y_train, X_valid, y_valid, learning_rate=0.01, epochs=2500):
        np.random.seed(1)  # setting a seed value
        self.n = X_train.shape[0]  # size of training set
        self.layers_size.insert(0, X_train.shape[1])  # the input dimension will decide the weights/bias dimensions
        self.init_model()  # initializing weights and bias

        for loop in range(epochs):  # for a certain number of epochs
            a, store = self.forward(X_train)  # do a forward pass
            loss = -np.mean(y_train * np.log(a.T + 1e-8))  # calculate loss
            derivatives = self.backward(X_train, y_train, store)  # calculate derivatives

            for layer_index in range(1, self.L + 1):  # weight updation
                self.parameters["W" + str(layer_index)] = self.parameters["W" + str(layer_index)] - learning_rate * \
                                                          derivatives[
                                                              "dW" + str(layer_index)]
                self.parameters["b" + str(layer_index)] = self.parameters["b" + str(layer_index)] - learning_rate * \
                                                          derivatives[
                                                              "db" + str(layer_index)]

            if loop % 10 == 0:  # logging of loss and accuracy after every few epochs
                a_val, _ = self.forward(X_valid)
                val_loss = -np.mean(y_valid * np.log(a_val.T + 1e-8)).round(3)
                train_acc = self.predict(X_train, y_train)
                val_acc = self.predict(X_valid, y_valid)
                print("Train Loss: ", loss.round(3), "Train Accuracy:", train_acc,
                      "Val Loss:", val_loss, "Val Accuracy:", val_acc)

                self.train_loss.append(loss)
                self.train_acc.append(train_acc)
                self.val_loss.append(val_loss)
                self.val_acc.append(val_acc)

    def predict(self, X, y):  # to calculate accuracy of train/validation
        a, _ = self.forward(X)
        y_hat = np.argmax(a, axis=0)
        y = np.argmax(y, axis=1)
        accuracy = (y_hat == y).mean()
        return (accuracy * 100).round(2)

    def plot_loss(self):  # plot the loss curve
        plt.figure()
        plt.plot(np.arange(len(self.train_loss)), self.train_loss, label="Train")
        plt.plot(np.arange(len(self.val_loss)), self.val_loss, label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.show()

    def plot_acc(self):  # plot the accuracy curve
        plt.figure()
        plt.plot(np.arange(len(self.train_acc)), self.train_acc, label="Train")
        plt.plot(np.arange(len(self.val_acc)), self.val_acc, label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.show()


if __name__ == '__main__':
    # hyperparameters
    hidden_nodes = 50
    lr = 0.1
    epochs = 1000

    # load the data
    all_train_x = pd.read_csv("train_data.csv").values  # normalized
    all_train_y = pd.read_csv("train_labels.csv").values  # one-hot encoded

    # 80-20 split of train-validation set
    X_train, X_valid, y_train, y_valid = train_test_split(all_train_x, all_train_y, test_size=0.2, random_state=42)

    mlp = MLP_Q4(hidden_nodes)  # initializing model class
    mlp.fit(X_train, y_train, X_valid, y_valid, learning_rate=lr, epochs=epochs)  # calling the fit function
    print("Train Accuracy:", mlp.predict(X_train, y_train))
    print("Validation Accuracy:", mlp.predict(X_valid, y_valid))

    # plotting accuracy and loss
    mlp.plot_loss()
    mlp.plot_acc()
