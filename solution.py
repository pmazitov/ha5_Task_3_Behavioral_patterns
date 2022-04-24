import random
import pickle


def get_random_vector(n):
    """Return `n` floats from -1 to 1."""
    return [random.random() * 2 - 1 for _ in range(n)]


class DumbNeuralNetwork():
    def __init__(self, weights_count):
        self.weights_count = weights_count
        self.weights = get_random_vector(self.weights_count)

    def train(self):
        random_vector = get_random_vector(self.weights_count)
        for i in range(self.weights_count):
            self.weights[i] += random_vector[i]

    def predict(self, X_test):
        return [
            sum(feature * coef for feature, coef in zip(x, self.weights))
            for x in X_test
        ]

    def evaluate(self, X_test, y_test):
        y_predicted = self.predict(X_test)
        loss_sum = sum(abs(y1 - y2) for y1, y2 in zip(y_predicted, y_test))
        loss_average = loss_sum / self.weights_count
        return loss_average

    def save_weights(self):
#         <YOUR CODE HERE>
        """
        Save weights to restore them later.
        """
        with open('dumb_weights.pickle', 'wb') as handle:
            pickle.dump(self.weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

    def restore_weights(self):
#         <YOUR CODE HERE>
        """
        Restore weights saved previously.
        """
        with open('dumb_weights.pickle', 'rb') as handle:
            self.weights = pickle.load(handle)


if __name__ == "__main__":
    weights_count = 1000
    dumb_NN = DumbNeuralNetwork(weights_count)

    test_samples = 100
    # [[<float -1..1>, ...], ...]
    X_test = [get_random_vector(weights_count) for _ in range(test_samples)]
    # [<float -2000..2000>, ...]
    y_test = [value * weights_count * 2 for value in get_random_vector(test_samples)]

    epoch_number = 10000
    best_loss = weights_count * 2

    for epoch in range(epoch_number):
        dumb_NN.save_weights()
        dumb_NN.train()

        loss = dumb_NN.evaluate(X_test, y_test)
        print(f'{epoch=}\t{loss=}\t{best_loss=}')

        if loss < best_loss:
            best_loss = loss
        else:
            print(f'rolling back...')
            dumb_NN.restore_weights()

    print(f'Result: {best_loss=}')
