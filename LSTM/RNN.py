import numpy as np
from neural_networks import NeuralNetwork
from layers import RNN, Activation
from losses import CrossEntropy
from optimizers import StochasticGradientDescent
from progress import to_categorical, train_test_split
from metric import accuracy_score

optimizer = StochasticGradientDescent()


def gen_mult_ser(nums):
    """ Method which generates multiplication series """
    X = np.zeros([nums, 10, 61], dtype=float)
    y = np.zeros([nums, 10, 61], dtype=float)
    for i in range(nums):
        start = np.random.randint(2, 7)
        mult_ser = np.linspace(start, start * 10, num=10, dtype=int)
        X[i] = to_categorical(mult_ser, n_col=61)
        y[i] = np.roll(X[i], -1, axis=0)
    y[:, -1, 1] = 1  # Mark endpoint as 1
    return X, y


def gen_num_seq(nums):
    """ Method which generates sequence of numbers """
    X = np.zeros([nums, 10, 20], dtype=float)
    y = np.zeros([nums, 10, 20], dtype=float)
    for i in range(nums):
        start = np.random.randint(0, 10)
        num_seq = np.arange(start, start + 10)
        X[i] = to_categorical(num_seq, n_col=20)
        y[i] = np.roll(X[i], -1, axis=0)
    y[:, -1, 1] = 1  # Mark endpoint as 1
    return X, y


if __name__ == '__main__':
    X, y = gen_mult_ser(3000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    clf = NeuralNetwork(optimizer=optimizer, loss=CrossEntropy)
    clf.add(RNN(10, activation="tanh", bptt_trunc=5, input_shape=(10, 61)))
    clf.add(Activation('softmax'))

    tmp_X = np.argmax(X_train[0], axis=1)
    tmp_y = np.argmax(y_train[0], axis=1)
    print("Number Series Problem:")
    print("X = [" + " ".join(tmp_X.astype("str")) + "]")
    print("y = [" + " ".join(tmp_y.astype("str")) + "]")
    print()
    train_err, _ = clf.fit(X_train, y_train, n_epochs=500, batch_size=512)
    y_pred = np.argmax(clf.predict(X_test), axis=2)
    y_test = np.argmax(y_test, axis=2)
    accuracy = np.mean(accuracy_score(y_test, y_pred))
    print(accuracy)

    print()
    print("Results:")
    for i in range(5):
        tmp_X = np.argmax(X_test[i], axis=1)
        tmp_y1 = y_test[i]
        tmp_y2 = y_pred[i]
        print("X      = [" + " ".join(tmp_X.astype("str")) + "]")
        print("y_true = [" + " ".join(tmp_y1.astype("str")) + "]")
        print("y_pred = [" + " ".join(tmp_y2.astype("str")) + "]")
        print()
