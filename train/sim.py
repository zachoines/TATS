import sys
import time
import math
from random import seed
from random import random
import numpy as np

import PyQt5
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPainter, QPixmap, QImage
from PyQt5.QtCore import (QObject, QPointF, QPropertyAnimation, pyqtProperty)
from PyQt5.QtGui import QPixmap


from time import sleep
import numpy as np

class FaceImage(QObject):
    def __init__(self):
        super().__init__()
        
        # Create image
        self.image = QImage()
        self.image.load("face.jpg")
        self.pixmap = QPixmap.fromImage(self.image)
        self.faceImage = QGraphicsPixmapItem(self.pixmap) 
        self.faceImage.setFlag(QGraphicsPixmapItem.ItemIsMovable) 

        self._default_size = (self.pixmap.height(), self.pixmap.width())
    
    def getHeight(self):
        return self.pixmap.height()

    def getWidth(self):
        return self.pixmap.width()
    
    def getX(self):
        return self.faceImage.x()

    def getY(self):
        return self.faceImage.y()
    
    def getPixmap(self):
        return self.faceImage

    def getSize(self):
        return self._default_size
    

    @pyqtProperty(float)
    def size(self):
        return (self.getWidth(), self.getHeight())

    def _set_pos(self, pos):
        self.faceImage.setPos(pos)

    def _set_size(self, size):
        self.pixmap = QPixmap.fromImage(self.image)
        self.pixmap = self.pixmap.scaledToWidth(size.width())
        self.faceImage.setPixmap(self.pixmap)

    pos = pyqtProperty(QPointF, fset=_set_pos) 
    size = pyqtProperty(PyQt5.QtCore.QSize, fset=_set_size) 

class Window(QMainWindow):
    def __init__(self, height, width):
        super().__init__()

        self.title = "PyQt5 Window"
        self.top = 100
        self.left = 100
        self.width = width
        self.height = height
        self.InitWindow()

    def start(self):
        self.start3DAnimationSession(1)

    def InitWindow(self):

        self.face = FaceImage()
        scene = QGraphicsScene()
        
        # Create a button
        self.startButton = QPushButton("Start Training", self)
        self.startButton.setGeometry(QRect(0, self.height - self.height / 9, 150, 50))
        # self.startButton.clicked.connect(self.start2DAnimationSession)
        self.startButton.clicked.connect(self.start)

        # Add image to screen
        scene.addItem(self.face.getPixmap())
        self.face._set_size(PyQt5.QtCore.QSize(self.face.getWidth() / 4, self.face.getHeight()))

        # Set the view
        self.view = QGraphicsView(scene, self)
        self.view.setStyleSheet("background:grey;")
        self.view.setGeometry(0, 0, self.width, self.height - self.height / 9)

        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)

        # Show Everything
        self.show()
    
    # Creates the illusion of depth, or z-axis movement, by scrinking size in proportion to a randomly generated geometric curve. Lasts for about 24 hours
    def start3DAnimationSession(self, step):

        maxStepsPerMinute = 1000
        minScale = 10
        runTimeScale = 6
        numMinutes = 60 * 24 * runTimeScale
        milliPerMinute = 60000
        
        # Turn off button for training session
        self.startButton.setEnabled(False)

        yScale = random()
        xScale = random()
        wScale = random()
        height = self.face.getHeight()
        width = self.face.getWidth()
        buttonHeight = self.startButton.size().height()
        newX = self.clamp(self.width * xScale - width, 0.0, self.width)
        newY = self.clamp(self.height * yScale - height, 0.0, self.height) - (buttonHeight * 2.0)
        newFaceWidth = max(self.face.getSize()[1] * wScale, self.face.getSize()[1] / minScale)

        # Bound to reasonable values
        newX = newX if newX >= 0 else 0
        newY = newY if newY >= 0 else 0


        currentWidth = self.face.getWidth()
        currentX = self.face.getX()
        currentY = self.face.getY()
        percentDifferenceWidth = abs(newFaceWidth - currentWidth) / currentWidth if currentWidth > 0 else 1.0
        
        newRatioX = abs(newX - currentX) / self.width 
        newRatioY = abs(newY - currentY) / self.height
        maxStepsPerMinute = maxStepsPerMinute * (newRatioX + newRatioY)

        ts = None
        xs = None
        ws = None
        ys = None
        chance = random()

        if chance <= (1/3):
            xs = np.linspace(start = currentX, stop = newX, num = int(maxStepsPerMinute / 100))
            ys = np.linspace(start = currentY, stop = newY, num = int(maxStepsPerMinute / 100))
            ws = np.linspace(start = currentWidth, stop = newFaceWidth, num = int(maxStepsPerMinute / 100))
            ts = np.linspace(start = 0.01, stop = 1, num = int(maxStepsPerMinute / 100), endpoint=True)
        elif chance > (1/3) and chance < (2/3):
            xs = np.linspace(start = currentX, stop = newX, num = int(maxStepsPerMinute))
            ys = np.linspace(start = currentY, stop = newY, num = int(maxStepsPerMinute))
            ws = np.linspace(start = currentWidth, stop = newFaceWidth, num = int(maxStepsPerMinute))
            ts = (1 - np.geomspace(start = 0.01, stop =  1, num = int(maxStepsPerMinute), endpoint=True))
            ts = reversed(ts)
        else:
            xs = np.linspace(start = currentX, stop = newX, num = int(maxStepsPerMinute))
            ys = np.linspace(start = currentY, stop = newY, num = int(maxStepsPerMinute))
            ws = np.linspace(start = currentWidth, stop = newFaceWidth, num = int(maxStepsPerMinute))
            ts = np.geomspace(start = 0.01, stop =  1, num = int(maxStepsPerMinute), endpoint=True)
        
        xs = np.linspace(start = currentX, stop = newX, num = int(maxStepsPerMinute))
        ys = np.linspace(start = currentY, stop = newY, num = int(maxStepsPerMinute))
        ws = np.linspace(start = currentWidth, stop = newFaceWidth, num = int(maxStepsPerMinute))
        hs = np.linspace(start = self.face.getWidth(), stop = newFaceWidth, num = int(maxStepsPerMinute))

        self.anim = QPropertyAnimation(self.face, b'pos')
        self.anim2 = QPropertyAnimation(self.face, b'size')
        self.anim.setDuration(milliPerMinute / runTimeScale)
        self.anim2.setDuration(milliPerMinute / runTimeScale)
        self.anim.setStartValue(QPointF(currentX, currentY))
        self.anim2.setStartValue(PyQt5.QtCore.QSize(currentWidth, 0.0))
        self.anim.setEndValue(QPointF(newX, newY))
        self.anim2.setEndValue(PyQt5.QtCore.QSize(newFaceWidth, 0.0))
        
        for (t, x, y, w) in zip(ts, xs, ys, ws):
            self.anim.setKeyValueAt(t, QPointF( x, y))
            self.anim2.setKeyValueAt(t, PyQt5.QtCore.QSize( w, .0))

            
        self.anim.start()
        self.anim2.start()

        if (step <= numMinutes):
            QTimer.singleShot(milliPerMinute / runTimeScale, lambda: self.start3DAnimationSession(step + 1))

    def clamp(self, x, minX, maxX):
        return max(min(x, maxX), minX)

    # Creates random, smooth, transitions in the x and y axis. Optionally can randomize image size as well. Lasts for 24 hours
    def start2DAnimationSession(self, image_size_animation=True, num_hours = 24, steps_per_minute = 20  ):
        maxStepsPerMinute = steps_per_minute
        minScale = 2
        numMinutes = 60 * num_hours # 24 hours
        milliseconds = 60000 
        maxAnimationDuration = milliseconds * numMinutes 
        numStepsInAnimation = maxStepsPerMinute * numMinutes

        # Turn off button for training session
        self.startButton.setEnabled(False)
        
        self.anim = QPropertyAnimation(self.face, b'pos')
        self.anim.setDuration(maxAnimationDuration)
        self.anim.setStartValue(QPointF(0, 0))

        if image_size_animation:
            self.anim2.setDuration(maxAnimationDuration)
            self.anim2 = QPropertyAnimation(self.face, b'size')
            self.anim2.setStartValue(PyQt5.QtCore.QSize(self.face.getWidth(),self.face.getHeight()))

        stepsInHour = numStepsInAnimation / 24
        totalAnnealSteps = stepsInHour * 1.0
        currentSteps = 0
        for step in range(numStepsInAnimation):
            
            yScale = np.random.uniform()
            xScale = np.random.uniform()
            hScale = np.random.uniform()

            # Uncommon if you want the face to hover around center for a while
            # if currentSteps < totalAnnealSteps:
            #     currentSteps += 1
            #     ratio = (currentSteps / totalAnnealSteps)
            #     chance = np.random.uniform() < ratio
            #     if chance: 
            #         yScale = np.random.uniform()
            #         xScale = np.random.uniform()
            #         hScale = np.random.uniform()
            #     else:
            #         yScale = self.clamp(np.random.normal(.5, 0.2), 0.0, 1.0)
            #         xScale = self.clamp(np.random.normal(.5, 0.2), 0.0, 1.0)
            #         hScale = self.clamp(np.random.normal(.5, 0.2), 0.0, 1.0)
            # else:
            #     yScale = np.random.uniform()
            #     xScale = np.random.uniform()
            #     hScale = np.random.uniform()

            newX = self.width * xScale - self.face.getWidth()
            buttonHeight = self.startButton.size().height()
            newY = (self.height * yScale - self.face.getHeight()) - buttonHeight * 2
            newFaceWidth = self.face.getWidth() * hScale
            newFaceHeight = self.face.getHeight() * yScale

            self.anim.setKeyValueAt(step / (maxStepsPerMinute * numMinutes), QPointF( newX if newX >= 0 else 0, newY if newY >= 0 else 0))

            if image_size_animation:
                self.anim2.setKeyValueAt(step / (maxStepsPerMinute * numMinutes), PyQt5.QtCore.QSize( newFaceWidth if newFaceWidth >= self.face.getWidth() / minScale else self.face.getWidth() / minScale, newFaceHeight if newFaceHeight >= self.face.getHeight() / minScale else self.face.getHeight() / minScale))
        
        self.anim.start()

        if image_size_animation:
            self.anim2.start()
        
        QTimer.singleShot(numStepsInAnimation, lambda: self.startButton.setDisabled(False))

def main():
    seed(time.time() * 1000)
    App = QApplication(sys.argv)
    screen = App.primaryScreen()
    size = screen.size()
    window = Window(size.height(), size.width())

    sys.exit(App.exec())

if __name__ == "__main__":
    main()