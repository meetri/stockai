import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from lib.lstm import LSTM


class LstmModel():

    def __init__(self, **kwargs):  # epochs=200, lr=0.001, gpu_device=None):

        # FIXME: Bug with loss_fn.backward() when using "mps" device
        self.gpu_device = kwargs.get("gpu_device")

        self.gpu_device = "cpu"
        self.device = torch.device(self.gpu_device or "cpu")

        self.epochs = kwargs.get("epochs", 200)

        self.learning_rate = kwargs.get("lr", 0.001)

        self.train_on = kwargs.get(
            "train_on", ["Open", "High", "Low", "Close", "Volume"])

        self.targets = kwargs.get("target", ["Close"])

        # number of features
        self.input_size = len(self.train_on) - len(self.targets)
        self.hidden_size = kwargs.get("hidden_size", 40)

        self.num_training_seq = kwargs.get("num_training_seq", 100)
        self.num_prediction_seq = kwargs.get("num_prediction_seq", 50)
        self.percent_for_training = kwargs.get("percent_for_training", 0.9)

        self.num_layers = kwargs.get("num_layers", 1)
        self.num_classes = kwargs.get("num_classes", self.num_prediction_seq)

        self.total_samples = None
        self.test_samples = None
        self.train_samples = None

        # fitted training data full dataset
        self.X_train = None
        self.y_train = None

        self.train_dataset = None
        self.test_dataset = None

        self.trainloader = None
        self.testloader = None

        self.raw_data = None
        self.mm = MinMaxScaler()
        self.ss = StandardScaler()
        self.rs = RobustScaler()
        self.lstm = None
        self.loss_fn = None
        self.optimizer = None

    def load(
        self, data=None, csvpath: str = None, start_at: int = 0,
        date_index=None
    ):

        if data is not None:
            self.raw_data = data
        elif csvpath.endswith("csv"):
            self.raw_data = pd.read_csv(
                csvpath, index_col='Date', parse_dates=True)
        elif csvpath.endswith("json"):
            self.raw_data = pd.read_json(
                csvpath)

        if date_index:
            self.raw_data[date_index] = (
                pd.to_datetime(
                    self.raw_data[date_index], unit="ms", origin="unix")
            )
            print(self.raw_data.columns)
            self.raw_data.set_index(date_index, inplace=True)

        self.raw_data = self.raw_data[self.train_on][start_at:]

        # calculate train / test sample count
        self.total_samples = self.raw_data.shape[0]

        self.train_samples = round(
            self.total_samples * self.percent_for_training)
        self.test_samples = self.total_samples - self.train_samples

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
