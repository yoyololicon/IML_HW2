import pandas as pd
import sys
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: %s <train file> <test file>' % sys.argv[0])
        sys.exit(1)
    train_data = pd.read_csv(sys.argv[1]).values
    test_data = pd.read_csv(sys.argv[2]).values
    [train_index, train_name, train_value, train_class] = np.split(train_data, [1, 2, -1], axis=1)
    [test_index, test_name, test_value, test_class] = np.split(test_data, [1, 2, -1], axis=1)
    total_train = np.squeeze(train_index[-1]+1)
    total_test = np.squeeze(test_index[-1]+1)
    print total_test, total_train

