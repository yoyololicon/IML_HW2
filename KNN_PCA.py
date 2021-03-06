import pandas as pd
import sys
import numpy as np
import KNN_classifier as KNN

classname = ['cp', 'im', 'pp', 'imU', 'om', 'omL', 'inL', 'imS']

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

    class_ind = [np.argwhere(train_class == c) for c in classname]

    #compute covariance
    cov = np.cov(train_value.T.astype(float))
    #compute the eigenvetor and sort them based on eigenvalue
    va, ve = np.linalg.eig(cov)
    eig_pairs = [(va[i], ve[:, i]) for i in range(len(va))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    accuracy = []
    best_k5 = []
    for i in range(len(eig_pairs)):
        #using different number of eigenvector
        W = np.empty([i+1, D])
        for j in range(i+1):
            W[j] = eig_pairs[j][1]

        tranformed = W.dot(train_value.T).T

        root = KNN.KDTree(np.concatenate((train_index, tranformed), axis=1), tranformed.shape[1])

        tranformed_t = W.dot(test_value.T).T

        alist = []
        for N in Test_N:
            ind = []
            for i in range(total_test):
                result = KNN.KNN_hyperplane(root, tranformed_t[i], N)
                ind.append(result)
            alist.append(KNN.accuracy(train_class[ind], test_class))
        accuracy.append(np.mean(alist))
        best_k5.append(alist[1])

    print ('K = 5, KNN_PCA accuracy: %f' % best_k5[np.argmax(accuracy)])

if __name__ == '__main__':
    main()