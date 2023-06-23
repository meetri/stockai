import numpy as np
from torch.utils.data import Dataset
import torch


class SequenceDataset(Dataset):

    def __init__(self, df):
        self.data = df

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.Tensor(sample['sequence']), torch.Tensor(sample['target'])

    def __len__(self):
        return len(self.data)


def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    "split a multivariate sequence past, future samples (X and y)"

    X, y = list(), list()
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences):
            break

        # gather input and output of the pattern
        seq_x, seq_y = (
            input_sequences[i:end_ix],
            output_sequence[end_ix-1:out_end_ix, -1]
        )
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)


def x_split_sequences(
    input_sequences, output_sequence, n_steps_in, n_steps_out
):
    "split a multivariate sequence past, future samples (X and y)"

    data = list()
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences):
            break
        # gather input and output of the pattern
        seq_x, seq_y = (
            input_sequences[i:end_ix],
            output_sequence[end_ix-1:out_end_ix, -1]
        )

        data.append({
            "sequence": seq_x,
            "target": seq_y
        })

    return data
