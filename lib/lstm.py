import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):

    def __init__(
        self, num_classes, input_size, hidden_size, num_layers,
        dropout=0, gpu_device=None
    ):
        super().__init__()
        self.num_classes = num_classes  # output size
        self.num_layers = num_layers  # number of recurrent layers in the lstm
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # neurons in each lstm layer

        self.device = torch.device(gpu_device or "cpu")

        # LSTM model
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, dropout=dropout)

        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected
        self.fc_2 = nn.Linear(128, num_classes)  # fully connected last layer
        self.relu = nn.ReLU()

    def forward(self, x):
        # hidden state
        h_0 = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        )

        # cell state
        c_0 = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        )

        # propagate input through LSTM
        output, (hn, cn) = (
            self.lstm(x, (h_0, c_0))
        )

        hn = hn.view(
            -1, self.hidden_size
        )  # reshaping the data for Dense layer next

        out = self.relu(hn)
        out = self.fc_1(out)  # first dense
        out = self.relu(out)  # relu
        out = self.fc_2(out)  # final output
        return out
