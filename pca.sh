#!/usr/bin/env bash

pip2 install --user numpy
pip2 install --user pandas
pip2 install --user sklearn
python2 PCA.py "$1" "$2"