from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QGraphicsScene, QGraphicsPixmapItem, QGraphicsView
from PyQt5.QtGui import QImage, QPixmap
import cv2
 
 
class GraphicsView(QGraphicsView):
    """
    Class documentation goes here.
    """
    def __init__(self, parent=None):
        super(GraphicsView, self).__init__(parent)
        self.scale = 1
    
    def load_img(self, QIm, x_scroball, y_scroball, x_scroball_max, y_scroball_max):
        image_height, image_width, image_depth = QIm.shape
        QIm = QImage(QIm.data, image_width, image_height,
                     image_width * image_depth,
                     QImage.Format_RGB888)
        pix = QPixmap.fromImage(QIm)
        self.item = QGraphicsPixmapItem(pix)
        self.scene = QGraphicsScene()
        self.scene.addItem(self.item)
        self.setScene(self.scene)
        self.item.setScale(self.scale)
        if x_scroball_max != 0:
            self.horizontalScrollBar().setMaximum(x_scroball_max)
            self.verticalScrollBar().setMaximum(y_scroball_max)
        self.horizontalScrollBar().setValue(x_scroball)
        self.verticalScrollBar().setValue(y_scroball)

    def get_scroball(self):
        return self.horizontalScrollBar().value(), self.verticalScrollBar().value()
    
    def get_max_scroball(self):
        return self.horizontalScrollBar().maximum(), self.verticalScrollBar().maximum()

    # def mousePressEvent(self, event):
    #     if event.buttons() == QtCore.Qt.LeftButton:
    #         self.scale = self.scale + 0.05
    #         if self.scale > 1.2:
    #             self.scale = 1.2
    #     elif event.buttons() == QtCore.Qt.RightButton:
    #         if self.scale <= 0:
    #             self.scale = 0.2
    #         else:
    #             self.scale = self.scale - 0.05
    #     self.item.setScale(self.scale)
    
    def wheelEvent(self, event):
        angle = event.angleDelta() / 8
        if angle.y() > 0:
            self.scale = self.scale + 0.05
            if self.scale > 10:
                self.scale = 10
        else:
            if self.scale <= 0:
                self.scale = 0.2
            else:
                self.scale = self.scale - 0.1
        self.item.setScale(self.scale)
 
def main():
    import sys
    app = QApplication(sys.argv)
    piczoom = picturezoom()
    piczoom.show()
    app.exec_()
 
if __name__ == '__main__':
    main()