#!/usr/bin/python
import cv2
import sys
import time
import rospy
import numpy
from cv_bridge import CvBridge
from open_face_recognition.srv import *


def main():
	rospy.init_node('compare_images')

	bridge = CvBridge()
	calc_feature_service = rospy.ServiceProxy('/calc_image_features', CalcImageFeatures)
	compare_images_service = rospy.ServiceProxy('/compare_images', CompareImages)

	image1 = cv2.imread(sys.argv[1])
	image2 = cv2.imread(sys.argv[2])

	msg1 = bridge.cv2_to_imgmsg(image1)
	msg2 = bridge.cv2_to_imgmsg(image2)

	# call the feature extraction service and calculate the distance by yourself
	t0 = time.time()
	features1 = numpy.array(calc_feature_service(msg1).features.data)
	t1 = time.time()
	features2 = numpy.array(calc_feature_service(msg2).features.data)
	t2 = time.time()

	print 'dim  :', len(features1)
	print 'img1 :', t1 - t0, '[sec]'
	print 'img2 :', t2 - t1, '[sec]'
	print 'total:', t2 - t0, '[sec]'

	if len(features1) < 5 or len(features2) < 5:
		self.setWindowTitle('failed to detect face landmarks!!')
		return

	dist = numpy.linalg.norm(features1 - features2)
	print 'distance', dist

	# call the image compare service to directly obtain the distance between images
	req = CompareImagesRequest()
	req.lhs = msg1
	req.rhs = msg2

	t0 = time.time()
	res = compare_images_service(req)
	t1 = time.time()

	print 'compare :', t1 - t0, '[sec]'
	print 'distance:', res.result

if __name__ == '__main__':
	main()
