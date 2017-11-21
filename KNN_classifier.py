import pandas as pd
import sys
import numpy as np
from collections import namedtuple
from pprint import pformat
import sklearn_KNN

def euclidean_dist(a, b):
    return np.sqrt(np.sum((a-b) ** 2))

class Node(namedtuple('Node', 'location index left_child right_child')):
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
            location=point_list[median],
            index=axis,
            left_child=KDTree(point_list[:median], D, depth + 1),
            right_child=KDTree(point_list[median + 1:], D, depth + 1)
        )

def KNN_search(rt, stack, value, k=1):
    node = rt
    while 1:
        axis = node.index
        if value[axis - 1] >= node.location[axis]:
            if node.right_child is not None:
                stack.append(node)
                node = node.right_child
            else:
                if node.left_child is not None:
                    stack.append(node)
                    node = node.left_child
                return node
        else:
            if node.left_child is not None:
                stack.append(node)
                node = node.left_child
            else:
                if node.right_child is not None:
                    stack.append(node)
                    node = node.right_child
                return node

def KNN_hyperplane(root, value):
    parents = []
    eliminate = []
    node = KNN_search(root, parents, value)
    best = (euclidean_dist(value, node.location[1:]), node.location[0])
    eliminate.append(node)

    while len(parents) > 0:
        p = parents.pop()
        axis = p.index
        current = euclidean_dist(value, p.location[1:])
        if current < best[0]:
            best = (current, p.location[0])
        #eliminate.append(p)

        if abs(value[axis-1] - p.location[axis]) <= best:
            #print value.shape, p.left_child, axis
            if value[axis-1] >= p.location[axis] and p.left_child is not None:
                node = KNN_search(p.left_child, parents, value)
                current = euclidean_dist(value, node.location[1:])
                if current < best[0]:
                    best = (current, node.location[0])
                #eliminate.append(node)
            elif p.right_child is not None:
                node = KNN_search(p.right_child, parents, value)
                current = euclidean_dist(value, node.location[1:])
                if current < best[0]:
                    best = (current, node.location[0])
                #eliminate.append(node)

    return best

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

    root = KDTree(np.concatenate((train_index, train_value), axis=1), train_value.shape[1])
    #print root

    for N in Test_N[:1]:
        ind = []
        for i in range(total_test):
            result = KNN_hyperplane(root, test_value[i])
            ind.append([result[1]])

        print 'KNN accuracy: ', sklearn_KNN.accuracy(train_class[ind], test_class)
        for i in range(3):
            print ' '.join(map(str, ind[i]))
        print ''


if __name__ == '__main__':
    main()

