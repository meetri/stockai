import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from lib.predict import split_sequences
import torch
# import torch.nn as nn
from torch.autograd import Variable
from lib.lstm import LSTM


class StockForecaster:

    def __init__(self):
        self.epochs = 1000
        self.learning_rate = 0.001  # 0.001 lr

        self.input_size = 4  # number of features
        self.hidden_size = 2  # number of features in hidden state
        self.num_layers = 1  # number of stacked lstm layers
        self.num_classes = 50  # number of output classes

        self.num_training_sequences = 100
        self.num_predictions = 50
        self.percent_for_training = 0.9

        self.total_samples = None
        self.test_samples = None
        self.train_samples = None

        # fitted training data full dataset
        self.X_train = None
        self.y_train = None

        self.X_ss = None
        self.y_mm = None

        self.X_train_tensors = None
        self.X_test_tensors = None

        self.y_train_tensors = None
        self.y_test_tensors = None

        self.raw_data = None
        self.mm = MinMaxScaler()
        self.ss = StandardScaler()
        self.lstm = None
        self.loss_fn = None
        self.optimiser = None

    def load(self, csvpath: str = "spydata.csv"):
        self.raw_data = pd.read_csv(
            csvpath, index_col='Date', parse_dates=True)
        self.raw_data.drop(columns=['Adj Close'], inplace=True)

        # calculate train / test sample count
        self.total_samples = len(self.raw_data.Close.values)
        self.train_samples = round(
            self.total_samples * self.percent_for_training)
        self.test_samples = self.total_samples - self.train_samples

    def normalize_data(self):
        # X contains all columns except `Close`
        # Y contains only `Close`
        X, y = (
            self.raw_data.drop(columns=['Close']),
            self.raw_data.Close.values.reshape(-1, 1)
        )

        self.X_train = self.ss.fit_transform(X)
        self.y_train = self.mm.fit_transform(y)

        p = self.prepare()
        self.create_tensors(
            X_train=p[0],
            X_test=p[1],
            y_train=p[2],
            y_test=p[3]
        )

    def prepare(self):
        # prepare data for training...
        self.X_ss, self.y_mm = split_sequences(
            self.X_train, self.y_train,
            self.num_training_sequences, self.num_predictions
        )

        # split data into train / test sets
        X_train = self.X_ss[:-self.test_samples]
        X_test = self.X_ss[-self.test_samples:]

        y_train = self.y_mm[:-self.test_samples]
        y_test = self.y_mm[-self.test_samples:]

        return X_train, X_test, y_train, y_test

    def create_tensors(self, X_train, X_test, y_train, y_test):
        # create tensors
        self.X_train_tensors = Variable(torch.Tensor(X_train))
        self.X_test_tensors = Variable(torch.Tensor(X_test))

        self.y_train_tensors = Variable(torch.Tensor(y_train))
        self.y_test_tensors = Variable(torch.Tensor(y_test))

    def create_model(self):
        self.lstm = LSTM(
            num_classes=self.num_classes,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )

        # mean-squared error for regression
        self.loss_fn = torch.nn.MSELoss()
        self.optimiser = torch.optim.Adam(
            self.lstm.parameters(),
            lr=self.learning_rate
        )

    def train(self):
        """
        n_epochs, lstm, optimiser, loss_fn, X_train,
        y_train, X_test, y_test):
        """

        loss_data = []
        for epoch in range(self.epochs):
            self.lstm.train()
            outputs = self.lstm.forward(self.X_train_tensors)  # forward pass

            # calculate the gradient, manually setting to 0
            self.optimiser.zero_grad()

            # obtain the loss function
            loss = self.loss_fn(outputs, self.y_train_tensors)
            loss.backward()  # calculates the loss of the loss function
            self.optimiser.step()  # improve from loss, i.e backprop

            loss_data.append(loss.item())

            # test loss
            self.lstm.eval()
            test_preds = self.lstm(self.X_test_tensors)
            test_loss = self.loss_fn(test_preds, self.y_test_tensors)
            if epoch % 100 == 0:
                print(
                    f"Epoch: {epoch}, train loss: {loss.item()}, "
                    f"test loss: {test_loss.item()}"
                )
        return loss_data
