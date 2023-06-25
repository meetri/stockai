from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np

from lib.lstm_model import LstmModel
import torch


class SequenceDataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.tensors = []
        for idx in range(len(self.data)):
            self.tensors.append(
                (
                    torch.Tensor(self.data[idx]),
                    torch.Tensor(self.labels[idx])
                )
            )

    def __getitem__(self, idx):
        return self.tensors[idx]
        # return torch.Tensor(self.data[idx]), torch.Tensor(self.labels[idx])

    def __len__(self):
        return len(self.data)


class StockForecast(LstmModel):

    def normalize(self):
        # X = self.raw_data.drop(columns=self.targets)
        X = self.raw_data
        y = self.raw_data[self.targets]

        self.X_train = self.ss.fit_transform(X)
        self.y_train = self.mm.fit_transform(y)

    def sequence(self, batch_size=64, shuffle=False):
        data, labels = [], []

        start_seq = self.num_training_seq
        end_seq = self.X_train.shape[0] - self.num_prediction_seq
        for i in range(start_seq, end_seq):
            indexes = range(i-self.num_training_seq, i, 1)
            data.append(self.X_train[indexes])
            labels.append(self.y_train[i:i+self.num_prediction_seq])

        full_seq_data, full_seq_test = np.array(data), np.array(labels)
        train_data = (
            full_seq_data[:-self.test_samples],
            full_seq_test[:-self.test_samples]
        )

        test_data = (
            full_seq_data[-self.test_samples:],
            full_seq_test[-self.test_samples:]
        )

        self.train_dataset = SequenceDataset(train_data[0], train_data[1])
        self.test_dataset = SequenceDataset(test_data[0], test_data[1])

        self.trainloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=shuffle)
        self.testloader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=shuffle)
