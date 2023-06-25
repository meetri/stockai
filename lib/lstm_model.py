import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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

        self.target = kwargs.get("target", "Close")

        self.input_size = len(self.train_on) - 1  # number of features
        self.hidden_size = kwargs.get("hidden_size", 40)

        self.num_training_sequences = kwargs.get("training_sequences", 100)
        self.num_predictions = kwargs.get("num_predictions", 50)
        self.percent_for_training = kwargs.get("percent_for_training", 0.9)

        self.num_layers = kwargs.get("num_layers", 1)
        self.num_classes = kwargs.get("num_classes", self.num_predictions)

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

    def load(
        self, csvpath: str = "spydata.csv", start_at: int = 0,
        date_index=None
    ):

        if csvpath.endswith("csv"):
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
