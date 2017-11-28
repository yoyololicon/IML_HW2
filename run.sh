#!/usr/bin/env bash

python2 -c "import numpy" || pip2 install --user numpy
python2 -c "import pandas" || pip2 install --user pandas
python2 KNN_classifier.py $@
python2 KNN_PCA.py $@