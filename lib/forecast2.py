from torch.utils.data import DataLoader

import numpy as np

from lib.lstm_model import LstmModel
from lib.helpers import SequenceDataset


class StockForecast(LstmModel):

    def normalize(self):
        X = self.raw_data.drop(columns=self.targets)
        y = self.raw_data[self.targets]

        self.X_train = self.rs.fit_transform(X)
        self.y_train = self.mm.fit_transform(y)

    def sequence(self, batch_size=64, shuffle=False):
        data, labels = [], []

        start_seq = self.num_training_seq
        end_seq = self.total_samples - self.num_prediction_seq
        for i in range(start_seq, end_seq):
            indexes = range(i-self.num_training_seq, i, 1)

            # history
            data.append(self.X_train[indexes])

            # the future
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

    def train(self, debug_freq=10):
        loss_data = []
        test_loss_data = []

        for epoch in range(self.epochs):

            self.lstm.train()
            loss_avg, cnt = 0, 0
            for train_tensors, test_tensors in self.trainloader:
                outputs = self.lstm.forward(train_tensors)

                # calculate the gradient, manually setting to 0
                self.optimizer.zero_grad()

                # obtain the loss function
                loss = self.loss_fn(
                    outputs, test_tensors[:, :, 0]).to(self.device)
                loss.backward()  # calculates the loss of the loss function
                self.optimizer.step()  # improve from loss, i.e backprop

                loss_avg += loss.item()
                cnt += 1
            loss_data.append(loss_avg / cnt)

            self.lstm.eval()
            loss_avg, cnt = 0, 0
            for train_tensors, test_tensors in self.testloader:
                outputs = self.lstm.forward(train_tensors)

                # calculate the gradient, manually setting to 0
                self.optimizer.zero_grad()

                # obtain the loss function
                loss = self.loss_fn(
                    outputs, test_tensors[:, :, 0]).to(self.device)
                loss.backward()  # calculates the loss of the loss function
                self.optimizer.step()  # improve from loss, i.e backprop

                loss_avg += loss.item()
                cnt += 1

            test_loss_data.append(loss_avg / cnt)

            if epoch % debug_freq == 0 or epoch == self.epochs-1:
                print(
                    f"Epoch: {epoch}, train loss: {loss_data[-1]}, "
                    f"test loss: {test_loss_data[-1]}"
                )
        return loss_data, test_loss_data
