import pygame
from NaiveBayes import NaiveBayes
import os
import time

class NaiveBayesUI:
    """
        Class that allowes us to draw our own digits and let the classifier
        do it's magic on them
    """
    
    def init(self):
        """
            Initialisation stuff
        """
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption('Naive Bayes for digit recognition')
        screen_width=330
        screen_height=280
        self.screen=pygame.display.set_mode([screen_width,screen_height])                   # get the screen
        self.pixel = [[[(10,10,10), [i*10, j*10]] for i in range(28)] for j in range(28)]   # Create a matrix that has dimensions 28x28.
                                                                                            # Every entry is the rgb value of the pixel and
                                                                                            # its actual position on screen
        self.predicted = '?'
        pygame.font.init()
        self.res = pygame.font.Font(None, 36)                                               # Font for the predicted value
        self.text = pygame.font.Font(None, 18)                                              # Font for the normal text
        self.prev_state = pygame.mouse.get_pressed()
        
        self.classifier = NaiveBayes()                                                      # The classifier.
        PATH='./trained.pickle'

        if os.path.isfile(PATH):                                                            # If we trained it already we loade the training values
            self.classifier.train([],[],True)
        else:                                                                               # else we train it
            training = True
            
            print 'Reading MNIST'
            # first read the training data
            training_set, training_labels = self.classifier.read_MNIST(60000, training)
            print'DONE!\n'
            
            print 'Training'
            t = 's'
            start_time = time.localtime()
            self.classifier.train(training_set, training_labels, False) # train the classifier
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
        self.initialized = True
        
    def draw(self):
        """
            The draw method that gets called in the "Mainloop"
        """
        if self.initialized:
            self.screen.fill((0,0,0))
            for row in self.pixel:
                for pixel in row:
                    pygame.draw.rect(self.screen, pixel[0], (pixel[1][0], pixel[1][1], 10, 10))     # Draw the pixel in the matrix
            pygame.draw.rect(self.screen, (255, 0, 0), (280, 0, 50, 50))                            # Draws the predict "button"
            pygame.draw.rect(self.screen, (255,0,0),(280,230,50,50))                                # Draws the clar screen "button"
            pygame.draw.rect(self.screen, (255,255,255), (280, 0, 2, 280))                          # Border between picture and "buttons"
            text = self.res.render('= '+str(self.predicted), 1, (255, 255, 255))                    # Draws the wanted texts at their positions
            self.screen.blit(text, (285, 140))
            text = self.text.render('Clear', 1, (0,0,0))
            self.screen.blit(text, (285, 240))
            text = self.text.render('Screen', 1, (0,0,0))
            self.screen.blit(text, (285, 260))
            text = self.text.render('Predict', 1, (0,0,0))
            self.screen.blit(text, (285, 20))
            pygame.display.flip()
            
    def addtuples(self,x,y):
        """
            Helper method to simply add two RGB tupels
        """
        a = []
        for i in range(len(x)):
            b = x[i] + y[i]
            if b < 0:
                b = 0
            if b > 255:
                b = 255
            a.append(b)
        return tuple(a)
        
    def update(self):
        """
            The update function is called every time in the "Mainloop"
        """
        self.mouse_state = pygame.mouse.get_pressed()
        if self.mouse_state[0] == 1:                                            # if the LMB is pressed we eiter want to draw some pixel on the screen, ...
            pos = pygame.mouse.get_pos()
            if pos[0] >= 0 and pos[1] >= 0 and pos[0] < 280 and pos[1] < 280:
                x = pos[0] / 10 % 28
                y = pos[1] / 10 % 28
                self.pixel [y][x][0] = (255,255,255)
                if y > 0:
                    self.pixel[y-1][x][0] = self.addtuples(self.pixel[y-1][x][0], (5, 5, 5))
                if y < 27:
                    self.pixel[y+1][x][0] = self.addtuples(self.pixel[y+1][x][0], (5, 5, 5))
                if x > 0:
                    self.pixel[y][x-1][0] = self.addtuples(self.pixel[y][x-1][0], (5, 5, 5))
                if x < 27:
                    self.pixel[y][x+1][0] = self.addtuples(self.pixel[y][x+1][0], (5, 5, 5))
                    
            elif pos[0] >= 285 and pos[0] <= 330 and pos[1] >= 230 and pos[1] <= 280 and self.prev_state[0] != self.mouse_state[0]: # ... clicked the clear button, ...
                for i in range(28):
                    for j in range(28):
                        self.pixel[i][j][0] = (10,10,10)
                self.predicted = '?'
            elif pos[0] >= 285 and pos[0] <= 330 and pos[1] >= 0 and pos[1] <= 50 and self.prev_state[0] != self.mouse_state[0]: # ... or we want our picture to be predicted
                image = []
                for i in range(28):
                    for j in range(28):
                        image.append(self.pixel[i][j][0][0])
                self.predicted = self.classifier.predict(image)
                
        self.prev_state = self.mouse_state
            
            
    def main(self):
        """
            "Mainloop"
        """
        while 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            self.update()
            self.draw()
        
if __name__ == '__main__':
    nb = NaiveBayesUI()
    nb.init()
    nb.main()