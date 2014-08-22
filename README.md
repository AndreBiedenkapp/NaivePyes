NaivePyes
=========
Python Implementation for handwritten Digit recognition using naive Bayes.
Trained and tested with the MNIST data set.

## Requirements:
    - numpy
    - matplotlib
    - pygame
    - [MNIST files](http://yann.lecun.com/exdb/mnist/)

### NaiveBayes.py
    This is used to train the classifier and to predict with it.
    It outputs a confusion matrix.
 
### NaiveBayesUI.py
    This uses pygame to allow you to draw your own numbers.
    It will use NaiveBayes.py to classify your handwritten digit.
    NaiveBayesUI won't center your pictures, so the classifier
    will give the best results if you try to draw digits centered