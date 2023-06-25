from lib.lstm_model import LstmModel


class StockForecast(LstmModel):

    def normalize_data(self):
        X = self.raw_data.drop(columns=[self.target])
        y = self.raw_data[self.target].values.reshape(-1, 1)

        self.X_train = self.ss.fit_transform(X)
        self.y_train = self.mm.fit_transform(y)
