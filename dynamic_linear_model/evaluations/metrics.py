from sklearn.metrics import accuracy_score, mean_squared_error


class Metrics:
    def __int__(self):
        pass

    def accuracy(self, y_true, y_pred):
        return accuracy_score(list(y_pred), list(y_true))
    
    def mse(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)