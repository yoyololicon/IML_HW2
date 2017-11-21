import pandas as pd
import sys
import numpy as np
from collections import namedtuple
from operator import itemgetter
from pprint import pformat


class Node(namedtuple('Node', 'location left_child right_child')):
    def __repr__(self):
        return pformat(tuple(self))


def KDTree(point_list, D, depth=0):
    axis = depth % D + 1

    # Sort point list and choose median as pivot element
    point_list = point_list[point_list[:, axis].argsort()]
    median = len(point_list) // 2  # choose median

    # Create node and construct subtrees
    if len(point_list) > 0:
        return Node(
            location=point_list[median].tolist(),
            left_child=KDTree(point_list[:median], D, depth + 1),
            right_child=KDTree(point_list[median + 1:], D, depth + 1)
        )

def KNN_search(node, value, k=1, depth=0):
    axis = depth % len(value)
    if value[axis] >= node.location[axis+1]:
        if node.right_child is not None:
            return KNN_search(node.right_child, value, k, depth + 1)
        else:
            return node.location[0]
    else:
        if node.left_child is not None:
            return KNN_search(node.left_child, value, k, depth + 1)
        else:
            return node.location[0]

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

    root = KDTree(np.concatenate((train_index, train_value), axis=1), train_value.shape[1])
    #print root

    for N in Test_N[:1]:

        ind = KNN_search(root, test_value[0])
        print ind, test_value[0]
        """
        print 'KNN accuracy: ', accuracy(train_class[ind], test_class)
        for i in range(3):
            print ' '.join(map(str, ind[i]))
        print ''
        """

if __name__ == '__main__':
    main()

