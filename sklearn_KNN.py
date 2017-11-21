from sklearn.neighbors import KDTree
import numpy as np
import pandas as pd
import sys
from collections import Counter

def evaluate(nearests, label):
    if len(nearests.shape) < 1:
        t = nearests
    else:
        t = Counter(nearests).most_common(1)[0][0]
    if t == label:
        return True
    else:
        return False

def accuracy(x, t):
    m = len(x)
    v = 0
    for i in range(m):
        if evaluate(np.squeeze(x[i]), np.squeeze(t[i])):
            v+=1
    return float(v)/m

def main():
    if len(sys.argv) < 3:
        print('usage: %s <train file> <test file>' % sys.argv[0])
        sys.exit(1)

    train_data = pd.read_csv(sys.argv[1]).values
    test_data = pd.read_csv(sys.argv[2]).values
    [train_index, train_name, train_value, train_class] = np.split(train_data, [1, 2, -1], axis=1)
    [test_index, test_name, test_value, test_class] = np.split(test_data, [1, 2, -1], axis=1)
    total_train = np.squeeze(train_index[-1] + 1)
    total_test = np.squeeze(test_index[-1] + 1)
    Test_N = [1, 5, 10, 100]

    root = KDTree(train_value)

    for N in Test_N:
        dist, ind = root.query(test_value, k=N)
        print 'KNN accuracy: ', accuracy(train_class[ind], test_class)
        for i in range(3):
            print ' '.join(map(str, ind[i]))
        print ''


if __name__ == '__main__':
    main()