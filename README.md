## Environment

* Ubuntu 16.04 LTS
* python2.7.12(using Pycharm 2017.2.3)
* extra module: numpy, sklearn, pandas

## Usage

Run the following command, this will display the accuracy of KNN and the k nearest neighbor's index of first three test data using different K (1, 5, 10, 100).
```
./run.sh <train file.csv> <test file.csv>
```

For the PCA version of KNN, run the following command:
```
./pca.sh <train file.csv> <test file.csv>
```
It's output format is the same as original KNN, but with M dimension of data (M=1~D, D=original dimension) after PCA transform the original training data.
And when M=2, it will plot the data distribution of all class using the most contributive pair of eigenvalue.
![](pca_2d.png)
Last, it will show the best dimension to have the highest average accuracy above all K.

For detailed discussion in chinese, please refer to this [report](REPORT.md).