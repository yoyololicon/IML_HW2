import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import KNN_classifier as KNN
import sklearn_KNN

def main():
    if len(sys.argv) < 3:
        print('usage: %s <train file> <test file>' % sys.argv[0])
        sys.exit(1)

    train_data = pd.read_csv(sys.argv[1]).values
    test_data = pd.read_csv(sys.argv[2]).values
    [train_index, train_name, train_value, train_class] = np.split(train_data, [1, 2, -1], axis=1)
    [test_index, test_name, test_value, test_class] = np.split(test_data, [1, 2, -1], axis=1)
    total_train = train_index[-1] + 1
    total_test = test_index[-1] + 1
    Test_N = [1, 5, 10, 100]
    D = train_value.shape[1]

    cov = np.cov(train_value.T.astype(float))

    va, ve = np.linalg.eig(cov)
    eig_pairs = [(va[i], ve[:, i]) for i in range(len(va))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    for i in range(len(eig_pairs)-1):
        W = np.empty([i+1, D])
        for j in range(i+1):
            W[j] = eig_pairs[j][1]

        tranformed = W.dot(train_value.T).T

        if i == 1:
            plt.scatter(tranformed[:, 0], tranformed[:, 1])
            plt.show()

        root = KNN.KDTree(np.concatenate((train_index, tranformed), axis=1), tranformed.shape[1])

        tranformed_t = W.dot(test_value.T).T
        print ' "', i+1, 'Dimension"'
        for N in Test_N:
            ind = []
            for i in range(total_test):
                result = KNN.KNN_hyperplane(root, tranformed_t[i], N)
                ind.append(result)
            print 'KNN accuracy: ', sklearn_KNN.accuracy(train_class[ind], test_class)
            for i in range(3):
                print ' '.join(map(str, ind[i]))
            print ''


if __name__ == '__main__':
    main()