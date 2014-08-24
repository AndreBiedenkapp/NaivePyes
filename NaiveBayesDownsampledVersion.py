import numpy as np
import struct
import matplotlib.pyplot as plt
import datetime
import time
import pickle
import os

class NaiveBayes():
    """
        Class that realizes a naive Bayes classifier for handwritten digit recognition.
        To train this classifier the MNIST dataset is used.
    """
    #################################################################################################
    def read_MNIST(self, howMany, train=True):
        """
            Method to read the .idxY-ubyte files downloaded form http://yann.lecun.com/exdb/mnist/
            howMany ->  used to determine how many labels/images have to be loaded. This method is
                        setup so that it will always load from the first image to the howManyth.
            train ->    used to determine if the training or the testing data has to be loaded.
        """
        if train:
            images = open('train-images.idx3-ubyte', 'rb')
            labels = open('train-labels.idx1-ubyte', 'rb')
            print 'Reading Training Data'
        else:
            images = open('t10k-images.idx3-ubyte','rb')
            labels = open('t10k-labels.idx1-ubyte','rb')
            print 'Reading Test Data'
        
        # ----------------------------- Loading the Labels ------------------------------------
        # Reading the magic number and the number of items in the file
        print '\nReading labels:'
        magicNumber = labels.read(4)
        print 'Magic number: ', struct.unpack('>I', magicNumber)[0]
        numberOfItems = labels.read(4)
        print 'Number of Items in MNIST File: ', struct.unpack('>I', numberOfItems)[0]
        if howMany > struct.unpack('>I', numberOfItems)[0]:
            howMany = struct.unpack('>I', numberOfItems)[0]
        self.howMany = howMany
        print 'Number of Files to read: ', howMany
        
        # reading the labels, depending on howMany should be read. (Every byte is a label)
        lable_list = []
        byte = labels.read(1)
        while len(byte) > 0 and len(lable_list) < howMany:
            lable_list.append(struct.unpack('>B', byte)[0])
            byte = labels.read(1)
        labels.close()
        i = 10
        if howMany < 10:
            i = howMany / 2
        print 'First ' + str(i) + ' labels: ', lable_list[:i]
        self.lable_list = lable_list
        
        # ----------------------------- Loading the Images ------------------------------------
        # reading the magic number, number of items, number of rows and columns
        print '\nReading Images:'
        magicNumber = images.read(4)
        print 'Magic number: ', struct.unpack('>I', magicNumber)[0]
        numberOfItems = images.read(4)
        print 'Number of Items in MNIST File: ', struct.unpack('>I', numberOfItems)[0]
        numOfRows = images.read(4)
        print 'Number of rows: ', struct.unpack('>I', numOfRows)[0]
        numOfCols = images.read(4)
        print 'Number of columns: ', struct.unpack('>I', numOfCols)[0]
        self.numOfCols = struct.unpack('>I', numOfCols)[0]
        self.numOfRows = struct.unpack('>I', numOfRows)[0]
        
        print ''
        
        if howMany > 10000:
            blub = int(howMany/25)
        else:
            blub = int(howMany/10)
        
        # reading the images, depending on howMany. (Every Byte is a pixel)
        image_list = []
        for i in range(howMany):
            if i > 0 and i % blub == 0:
                print 'Loaded ' + str(i/float(howMany)*100) + '% of the images'
            image = []
            for j in range(struct.unpack('>I', numOfRows)[0]*struct.unpack('>I', numOfCols)[0]):
                x = struct.unpack('>B', images.read(1))[0]
                image.append(x)            
            image_list.append(image)
        images.close()
        
        print ''
        
        # printing an sample Image in ASCII-art
        sampleImage = image_list[0]
        print 'Example image (should resemble a 5 when reading the training data, a 7 for testing):'
        self.print_ascii_picture(sampleImage)
        self.image_list = image_list
        return image_list, lable_list
    #################################################################################################
    def downsample(self, images):
        """ 28x28 (original input) --> 14x14
	    estimator -> default to mean.
	    TODO: MUSS ERST JEDE IMAGE AUS IMAGES NEHMEN, DOWNSAMPLEN UND ALLE IN EINEN NEUEN ARRAY REIN= DOWNSAMPLEDIMAGES
        """
	downsampledimages = []
	for i in range(len(images)):
	  image = images[i]
          downsampledimage = []
	  for x in range(14):
	    for y in range(14):
              px1 = image[2*x*28+2*y]
              px2 = image[(2*x+1)*28+2*y]
	      px3 = image[2*x*28+(2*y+1)]
  	      px4 = image[(2*x+1)*28+(2*y+1)]
              downsampledimage.append((px1+px2+px3+px4)/4)
	  downsampledimages.append(downsampledimage)
	return downsampledimages
	
    #################################################################################################
    def train(self, images, labels, load):
        """
            Method to train the Classifier or load a former pickled file
            images ->   List of Lists. Every Sub-list is a feature vector. The feature vector contains
                        the grey values of every pixel per image.
            labels ->   List that contains the labels of the images. labels[i] = label for image i.
        """
         
        PATH='./trained.pickle'

        if os.path.isfile(PATH) and load:
            print 'Loading already existing training values from ' + PATH
            with open('trained.pickle') as f:
                self.classes, self.prClass, self.prPixelGivenClass = pickle.load(f)
        else:
            self.prClass = [0 for i in range(10)]
            self.classes = [i for i in range(10)]
            self.prPixelGivenClass = [[0 for i in range(14*14)] for j in range(10)]
            
            for i in range(len(labels)):
                self.prClass[labels[i]] += 1 # Count how many times a class appears in the labels list.
                for j in range(len(images[i])):
                    if images[i][j] < 100:
                        self.prPixelGivenClass[labels[i]][j] += 1   # For every class, count how many times
                                                                    # a pixel is black.
                        
            for i in range(len(self.prPixelGivenClass)):
                for j in range(len(self.prPixelGivenClass[i])):
                    self.prPixelGivenClass[i][j] /= float(self.prClass[i])  # Divide the count of black pixels
                                                                            # by the number of times a class
                                                                            # appears, to get a percentage.
                self.prClass[i] /= float(len(images))                       # Divide the number of times a class
                                                                            # appears, by the total number of classes
                                                                            # to get a percentage.
            
            print ''
            for i in range(len(self.prClass)):  # some useful output that shows the probability of each class.
                print 'Pr(C=' + str(i) + ') = ' + str(self.prClass[i])[:5]
                # print 'Probabilites of the individual pixel in this class:'           ""Commented because we now have
                # self.print_ascii_probabilities(self.prPixelGivenClass[i])             ""'heat-maps' for each image
                # print''
            print ''
            with open('trained.pickle', 'w') as f:
                pickle.dump([self.classes, self.prClass, self.prPixelGivenClass], f)
            
    #################################################################################################
    def predict(self, image):
        """
            Method to predict to which class a certain image belongs. Instead of multiplying the
            individual probabilities, we sum over the logarithm so the computer has no problem
            with float precision.
            image ->    The image that has to be classified. The image has to be a feature-vector with
                        14*14 entries, that represent the pixel grey values.
        """
        result = -1
        max = [0.0 for i in range(10)]
        # print 'Image to predict: '        "" Only needed for debugging
        # self.print_ascii_picture(image)
        for i in range(len(self.classes)):
            predict = 0.0
            for j in range(len(image)):
                if image[j] < 100:
                    predict += np.log(self.prPixelGivenClass[i][j]) # If the pixel is black, use its probability
                                                                    # that was learned in the train method
                else:
                    predict += np.log(1-self.prPixelGivenClass[i][j] + 0.00000000000001)    # else use the probability
                                                                                            # for a white pixel plus 'epsilon'
                                                                                            # epsilon is needed so we don't
                                                                                            # get 0 as the white pixel probability.
                                                                                            # Also this will make max[i] large if
                                                                                            # we predict for a white pixel when the
                                                                                            # class expects a black pixel at this
                                                                                            # position.
            predict *= np.log(self.prClass[i])  # multiply that with the class probability.
            max[i] = predict
        return np.argmin(max) # return the index of the minimum value.
    
    #################################################################################################
    def print_ascii_picture(self, sampleImage):
        """
            Helper-method that lets us print an image as ASCII-art. Mostly needed for debugging.
            Prints the grey values of each pixel.
            sampleImage ->  The image that has to be printed. The image has to be a feature-vector with
                            14*14 entries, that represent the pixel grey values.
        """
        a = ''
        for i in range(self.numOfCols*self.numOfRows):
	    if i%28== 0:
		print a
                a = ''
            else:
                b = str(sampleImage[i])
                if b == '0' or b == '1':
                    b = '    '
                elif len(b) == 1:
                    b = '   ' + b
                elif len(b) == 2:
                    b = '  ' + b
                elif len(b) == 3:
                    b = ' ' + b
                a = a + b + ' '
        print a
    
    #################################################################################################
    def print_ascii_probabilities(self, sampleImage):
        """
            Same as print_ascii_picture but instead of printing the grey values, this prints the
            probabilities calculated in Training.
            sampleImage ->  The image that has to be printed. The image has to be a feature-vector with
                            14*14 entries, that represent the probabilities of the pixel for the given
                            class.
        """
        a = ''
        for i in range(self.numOfCols*self.numOfRows):
            if i%28 == 0:
                print a
                a = ''
            else:
                b = str(sampleImage[i])
                if len(b) < 3:
                    b = b + '00'
                elif len(b) < 4:
                    b = b + '0'
                else:
                    b = b[:4]
                a = a + b + ' '
        print a
        
    #################################################################################################
    def confusion_matrix(self, real, guessed):
        """
            Used to print the confusion matrix to the console.
        """
        matrix = [[0 for i in range(10)] for j in range(10)]
        for i in range(len(real)):
            matrix[real[i]][guessed[i]] += 1
        for i in range(len(matrix)):
            matrix[i][i] = 0
        return matrix
        
        # a = '     0    1    2    3    4    5    6    7    8    9\n'
        # for i in range(10):
            # a += str(i) + ' '
            # for j in range(10):
                # b = str(matrix[j][i])
                # if len(b) == 1:
                    # b = '   ' + b
                # elif len(b) == 2:
                    # b = '  ' + b
                # elif len(b) == 3:
                    # b = ' ' + b
                # a += b + ' '
            # a = a + '\n'
        # print a
    
    #################################################################################################
    def plot_graphic(self, m, name):
        """
            This lets us create images that we desire as output. ("heat-maps" and the conf_matrix)
            m ->    matrix that either contains the probabilities of the pixel for the heat-maps
                    or this is the conf_matrix
            name -> String that tells us which name the image will have.
        """
        none_float = False
        if type(m[0][0]) != type(0.0):  # if we don't have a float in the matrix we need to create a normalized matrix
            none_float = True           # so the confusion matrix can use that to determine which collor should be used
            norm_conf = []              # for each entry.
            for i in m:
                a = 0
                tmp_arr = []
                a = sum(i, 0)
                for j in i:
                    if a == 0:
                        a = 1
                    tmp_arr.append(float(j)/float(a))
                norm_conf.append(tmp_arr)

        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)       # create a subplot so we can write "number of confusions" in the confusion matrix
        ax.set_aspect(1)
        if none_float:
            res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                        interpolation='nearest')
        else:                           # if we have a heat-map we can simply use the given matrix m
            res = ax.imshow(np.array(m), cmap=plt.cm.jet, 
                        interpolation='nearest')

        width = len(m)
        height = len(m[0])

        if none_float:                  # for the confusion matrix write the "number of confusions" for each
            for x in xrange(width):     # entry.
                for y in xrange(height):
                    ax.annotate(str(m[x][y]), xy=(y, x), 
                                horizontalalignment='center',
                                verticalalignment='center')

        cb = fig.colorbar(res)  # Show the legend/colorbar
        
        # set the axis for either the confusion matrix or the heat-maps
        if width > 10:
            plt.xticks(range(0, width, 2), range(0, width, 2))
        else:
            plt.xticks(range(0, width), range(0, width))
        if height > 10:
            plt.yticks(range(0, height, 2), range(0, height, 2))
        else:
            plt.yticks(range(0, height), range(0, height))
        
        # append the date to the name so we don't overwrite something
        # ts = time.time()
        # st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H.%M.%S')
        plt.savefig(str(name) + '.png', format='png') # save the figure
    
    #################################################################################################
    def featurevector_to_matrix(self, vector, width, height):
        """
            Helper-method to turn a feature-vector into a matrix.
            vector ->   list that contains the pixel values.
            width ->    integer that tells us the width of the resulting matrix
            height ->   integer that tells us the height of the resulting matrix
        """
        matrix = [[0 for i in range(width)] for i in range(height)]
        j = 0
        t = 0
        for i in range(len(vector)):
            if (i > 0 and i % width == 0):
                j += 1
                t = 0
            matrix [j][t] = vector[i]
            t += 1
        return matrix
    
#####################################################################################################
if __name__ == '__main__': # Main method
    training = True
    nb = NaiveBayes()
    
    print 'Reading MNIST'
    # first read the training data
    training_set, training_labels = nb.read_MNIST(60000, training)
    print ''
    # then the testing data
    test_set, test_labels = nb.read_MNIST(10000, not training)
    print'DONE!\n'
    print'Downsampling..'
    training_set = nb.downsample(training_set)
    test_set = nb.downsample(test_set)
    print 'Training'
    t = 's'
    start_time = time.localtime()
    nb.train(training_set, training_labels, False) # train the classifier
    end_time = time.localtime()
    # just stuff for the timing output
    b = end_time[4] - start_time[4]
    if b < 0:
        b = 60 + b
    t = str(b) + 'min '
    a = end_time[5] - start_time[5]
    if a < 0:
        a = 60 + a
    t += str(a) + 'sec'
    print 'DONE IN ' + t + '!\n'
    
    print 'Predicting'
    result = []
    start_time = time.localtime()
    for i in range(len(test_set)): # for each testing data predict which class the image belongs to.
        if i % ((len(test_set)/15) + 1) == 0:
            x = str(i/float(len(test_set)) * 100)
            print x + '% done'
        result.append(nb.predict(test_set[i]))
    end_time = time.localtime()
    # again just for the timing output
    b = end_time[4] - start_time[4]
    if b < 0:
        b = 60 + b
    t = str(b) + 'min '
    a = end_time[5] - start_time[5]
    if a < 0:
        a = 60 + a
    t += str(a) + 'sec'
    print 'DONE IN ' + t + '!\n'
    
    acc = 0
    for i in range(len(result)):
        if result[i] == test_labels[i]:
            acc += 1
    acc /= float(len(result)) # calculate the accuracy.
    print 'Accuracy = ' + str(acc * 100) + '%\n'
    m = nb.confusion_matrix(test_labels, result)
    
    # plotting stuff.
    print 'Plotting! Saving Pictures in the folder where you executed the script'
    print 'Plotting confusion matrix'
    nb.plot_graphic(m, 'conf_matrix')
    for i in range(len(nb.prPixelGivenClass)):
        print 'Plotting "heatmap" for number ' + str(i)
        nb.plot_graphic(nb.featurevector_to_matrix(nb.prPixelGivenClass[i], 14, 14), i)
    print 'DONE!'
