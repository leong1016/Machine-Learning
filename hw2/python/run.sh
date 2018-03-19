#!/bin/bash

pushd regression
python BGD.py ; python SGD.py
popd

pushd classification
python perceptron.py ; python votedperceptron.py ; python averagedperceptron.py
popd