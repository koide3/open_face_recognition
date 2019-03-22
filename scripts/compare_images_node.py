#!/usr/bin/python
import cv2
import rospy
import numpy
from cv_bridge import CvBridge
from open_face_recognition.srv import *
from PyQt4 import QtGui


class Widget(QtGui.QWidget):
	def __init__(self, app):
		super(Widget, self).__init__()
		self.app = app
		self.initUI()
		self.setAcceptDrops(True)

		self.image1 = None

	def initUI(self):
		self.setWindowTitle('drop face images!!')
		self.setGeometry(300, 300, 256, 256)

	def dragEnterEvent(self, e):
		if e.mimeData().hasUrls():
			e.accept()
		else:
			e.ignore()

	def dropEvent(self, e):
		for url in e.mimeData().urls():
			path = str(url.toString())[7:]
			image = cv2.imread(path)
			if image is None:
				self.setWindowTitle('failed to load image')
				print 'failed to load image'
				print path
				return

			if self.image1 is None:
				self.setWindowTitle('drop the next image!!')
				self.image1 = image
				cv2.imshow('image1', image)
				cv2.moveWindow('image1', 600, 30)
				cv2.waitKey(100)
			else:
				cv2.imshow('image2', image)
				cv2.moveWindow('image1', 900, 30)
				cv2.waitKey(100)
				self.compare(self.image1, image)

				self.image1 = None
				self.image2 = None

	def compare(self, image1, image2):
		bridge = CvBridge()
		calc_feature_service = rospy.ServiceProxy('/calc_image_features', CalcImageFeatures)

		msg1 = bridge.cv2_to_imgmsg(image1)
		msg2 = bridge.cv2_to_imgmsg(image2)

		features1 = numpy.array(calc_feature_service(msg1).features.data)
		features2 = numpy.array(calc_feature_service(msg2).features.data)
		if len(features1) < 5 or len(features2) < 5:
			self.setWindowTitle('failed to detect face landmarks!!')
			return

		dist = numpy.linalg.norm(features1 - features2)
		self.setWindowTitle('dist:%.4f' % dist)
		print 'distance', dist


def main():
	rospy.init_node('compare_images')

	app = QtGui.QApplication(sys.argv)
	widget = Widget(app)
	widget.show()
	app.exec_()

if __name__ == '__main__':
	main()
