import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from lib.lstm import LSTM
from lib.helpers import SequenceDataset, x_split_sequences, split_sequences


class StockForecaster:

    def __init__(self, **kwargs):  # epochs=200, lr=0.001, gpu_device=None):
        self.epochs = kwargs.get("epochs", 200)
        self.learning_rate = kwargs.get("lr", 0.001)

        # FIXME: Bug with loss_fn.backward() when using "mps" device
        self.gpu_device = kwargs.get("gpu_device")

        self.gpu_device = "cpu"
        self.device = torch.device(self.gpu_device or "cpu")

        self.train_on = kwargs.get(
            "train_on", ["Open", "High", "Low", "Close", "Volume"])

        self.target = kwargs.get("target", "Close")

        self.input_size = len(self.train_on) - 1  # number of features
        self.hidden_size = kwargs.get("hidden_size", 40)
        self.num_layers = kwargs.get("num_layers", 1)
        self.num_classes = kwargs.get("num_classes", 50)

        self.num_training_sequences = kwargs.get("training_sequences", 100)
        self.num_predictions = kwargs.get("num_predictions", 50)
        self.percent_for_training = kwargs.get("percent_for_training", 0.9)

        self.total_samples = None
        self.test_samples = None
        self.train_samples = None

        # fitted training data full dataset
        self.X_train = None
        self.y_train = None

        self.trainloader = None
        self.testloader = None

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
        self.optimizer = None

    def load(self, csvpath: str = "spydata.csv", start_at: int = 0):
        self.raw_data = pd.read_csv(
            csvpath, index_col='Date', parse_dates=True)

        self.raw_data = self.raw_data[self.train_on][start_at:]

        # calculate train / test sample count
        self.total_samples = len(self.raw_data[self.target].values)

        self.train_samples = round(
            self.total_samples * self.percent_for_training)
        self.test_samples = self.total_samples - self.train_samples

    def normalize_data(self):
        # X contains all columns except `Close` ( self.target )
        # Y contains only `Close`
        X, y = (
            self.raw_data.drop(columns=[self.target]),
            self.raw_data[self.target].values.reshape(-1, 1)
        )

        self.X_train = self.ss.fit_transform(X)
        self.y_train = self.mm.fit_transform(y)

    def prepare(self):
        # prepare data for training...

        self.normalize_data()

        self.X_ss, self.y_mm = split_sequences(
            self.X_train, self.y_train,
            self.num_training_sequences, self.num_predictions
        )

        self.create_tensors()

    def prepare2(self):
        # prepare data for training...

        self.normalize_data()

        sequences = x_split_sequences(
            self.X_train, self.y_train,
            self.num_training_sequences, self.num_predictions
        )

        dataset = SequenceDataset(sequences)

        train_len = int(len(dataset)*self.percent_for_training)
        lens = [train_len, len(dataset)-train_len]

        train_ds, test_ds = random_split(dataset, lens)

        self.trainloader = DataLoader(
            train_ds, batch_size=16, shuffle=True, drop_last=True
        )

        self.testloader = DataLoader(
            test_ds, batch_size=16, shuffle=True, drop_last=True
        )

    def create_tensors(self):
        # split data into train / test sets
        X_train = self.X_ss[:-self.test_samples]
        X_test = self.X_ss[-self.test_samples:]

        y_train = self.y_mm[:-self.test_samples]
        y_test = self.y_mm[-self.test_samples:]

        # create tensors
        self.X_train_tensors = Variable(torch.Tensor(X_train)).to(self.device)
        self.X_test_tensors = Variable(torch.Tensor(X_test)).to(self.device)

        self.y_train_tensors = Variable(torch.Tensor(y_train)).to(self.device)
        self.y_test_tensors = Variable(torch.Tensor(y_test)).to(self.device)

    def create_model(self):
        self.lstm = LSTM(
            num_classes=self.num_classes,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            gpu_device=self.gpu_device
        ).to(self.device)

        # mean-squared error for regression
        self.loss_fn = torch.nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.lstm.parameters(),
            lr=self.learning_rate
        )

    def train2(self):
        loss_data = []
        test_loss_data = []
        for epoch in range(self.epochs):
            self.lstm.train()
            for x, y in self.trainloader:
                print("x")
                self.optimizer.zero_grad()

                outputs = self.lstm(x).squeeze().to(self.device)
                loss = self.loss_fn(
                    outputs, self.y_train_tensors
                ).to(self.device)

                loss_data.append(loss.item())
                print(loss_data)

                loss.backward()
                self.optimizer.step()

            self.lstm.eval()
            for x, y in self.testloader:
                print("y")
                with torch.no_grad():
                    output = self.lstm(x)
                    error = self.loss_fn(output, y)
                    test_loss_data.append(error.item())

        return loss_data, test_loss_data

    def train(self):
        """
        n_epochs, lstm, optimizer, loss_fn, X_train,
        y_train, X_test, y_test):
        """

        loss_data = []
        test_loss_data = []

        for epoch in range(self.epochs):
            self.lstm.train()
            outputs = self.lstm.forward(self.X_train_tensors)

            # calculate the gradient, manually setting to 0
            self.optimizer.zero_grad()

            # obtain the loss function
            loss = self.loss_fn(outputs, self.y_train_tensors).to(self.device)
            loss.backward()  # calculates the loss of the loss function
            self.optimizer.step()  # improve from loss, i.e backprop

            loss_data.append(loss.item())

            # test loss
            self.lstm.eval()
            test_preds = self.lstm(self.X_test_tensors)
            test_loss = self.loss_fn(test_preds, self.y_test_tensors)
            test_loss_data.append(test_loss.item())

            if epoch % 100 == 0:
                print(
                    f"Epoch: {epoch}, train loss: {loss.item()}, "
                    f"test loss: {test_loss.item()}"
                )
        return loss_data, test_loss_data
