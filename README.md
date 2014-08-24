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
![](/10classesNoSubsampling/conf_matrix.png)
The accuracy of this classifier is 83.34%

### NaiveBayesTwoClasses.py
This basically works like NaiveBayes.py but this creates
five classifiers that can differentiate between two classes.
Classifier 1 differentiates between 0 and 1
Classifier 2 differentiates between 2 and 5
Classifier 3 differentiates between 3 and 4
Classifier 4 differentiates between 6 and 9
Classifier 5 differentiates between 7 and 8
This also outputs a confusion matrix
![](/TwoClassesNoSubsampling/conf_matrix_5lassifier.png)
The classifiers alone all have accuracies above 93%
but combined their accuracy is 81.06%

### NaiveBayesDownsampledVersion.py
![](/NaiveBayesDownsampledVersion/conf_matrix.png)
accuracy is sadly 82,43%

### NaiveBayesTwoClassesDownsampledVersion.py
![](/ NaiveBayesTwoClassesDownsampledVersion/conf_matrix_5lassifier.png)
accuracy is 79.74%
 
### NaiveBayesUI.py
This uses pygame to allow you to draw your own numbers.
It will use NaiveBayes.py to classify your handwritten digit.
NaiveBayesUI won't center your pictures, so the classifier
will give the best results if you try to draw digits centered
![](/NaiveBayesUI.png)
