#!/usr/bin/env python
import cv2
import dlib
import numpy
import rospy
import itertools
from cv_bridge import CvBridge
from sensor_msgs.msg import *
from open_face_recognition.srv import *


# ros node
class LiveRecognitionNode:
	# constructor
	def __init__(self):
		print 'init'
		self.cfg_callback()
		self.cv_bridge = CvBridge()
		self.detector = dlib.get_frontal_face_detector()

		self.feat_service = rospy.ServiceProxy('/calc_image_features', CalcImageFeatures)
		self.image_sub = rospy.Subscriber(rospy.get_param('~topic', '/image'), Image, self.image_callback, queue_size=1, buff_size=2**24)

		self.input_image = None
		self.registred_faces = []
		print 'done'

	# TODO: use dynamic reconfigure
	def cfg_callback(self):
		self.image_width = int(rospy.get_param('~image_width', '640'))
		self.threshold = float(rospy.get_param('~threshold', '0.8'))

	def image_callback(self, image_msg):
		self.input_image = image_msg

	def spin(self):
		if self.input_image is None:
			return

		image_msg, self.input_image = self.input_image, None
		image = self.cv_bridge.imgmsg_to_cv2(image_msg, 'bgr8')
		scale = float(self.image_width) / image.shape[1]
		image = cv2.resize(image, (self.image_width, int(image.shape[0] * scale)))

		face_rects = self.detector(image)
		face_images = [image[x.top():x.bottom(), x.left():x.right()] for x in face_rects]
		face_features = [self.extract_features(image, x) for x in face_images]

		associations = self.associate(face_features)

		for rect, associated in zip(face_rects, associations):
			color = (0, 255, 0) if associated[1] >= 0 else (0, 0, 255)
			text = 'ID:%d' % associated[1] if associated[1] >= 0 else 'unknown!'

			cv2.rectangle(image, (rect.left()-1, rect.top()-1), (rect.right(), rect.bottom()), color)
			cv2.rectangle(image, (rect.left()-1, rect.top()-12), (rect.right(), rect.top()-1), color, -1)

			cv2.putText(image, text, (rect.left() + 3, rect.top() - 2), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255))
			cv2.putText(image, '%.3f' % associated[0], (rect.left() + 3, rect.top() + 12), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255))

		cv2.putText(image, 'press space to register a face!', (15, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 3)
		cv2.putText(image, 'press space to register a face!', (15, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))
		cv2.imshow('frame', image)

		if cv2.waitKey(5) & 0xFF == ord(' '):
			self.register_face(face_images, face_features)

	def extract_features(self, image, face_image):
		face_image = cv2.resize(face_image, (256, 256))

		req = CalcImageFeaturesRequest()
		req.image = self.cv_bridge.cv2_to_imgmsg(face_image)

		res = self.feat_service(req)
		if len(res.features.data) < 5:
			return None

		return numpy.array(res.features.data, dtype=numpy.float32)

	def register_face(self, face_images, face_features):
		if len(face_images) == 0:
			return

		largest = sorted(zip(face_images, face_features), key=lambda x: x[0].shape[0] * x[0].shape[1])[-1]
		face_image = cv2.resize(largest[0], (128, 128))
		face_id = len(self.registred_faces)

		cv2.putText(face_image, 'ID:%d' % face_id, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 3)
		cv2.putText(face_image, 'ID:%d' % face_id, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))

		cv2.putText(face_image, 'register?  [Y/n]', (10, 100), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 3)
		cv2.putText(face_image, 'register?  [Y/n]', (10, 100), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255))

		cv2.imshow('confirm', face_image)
		if cv2.waitKey(0) & 0xFF == ord('y') and largest[1] is not None:
			self.registred_faces.append(largest[1])
		cv2.destroyWindow('confirm')

	def associate(self, face_features):
		min_dists = [-1 for x in face_features]

		dists = []
		for f, r in itertools.product(enumerate(face_features), enumerate(self.registred_faces)):
			if f[1] is None:
				continue

			d = numpy.linalg.norm(f[1] - r[1])
			min_dists[f[0]] = min_dists[f[0]] if min_dists[f[0]] >= 0.0 and min_dists[f[0]] < d else d

			if d < self.threshold:
				dists.append((d, f[0], r[0]))

		dists = sorted(dists, key=lambda x: x[0])

		established = {}
		while len(dists):
			front = dists[0]
			established[front[1]] = front[2]
			dists = [x for x in dists if x[1] != front[1] and x[2] != front[2]]

		return [(min_dists[x], established[x]) if x in established else (min_dists[x], -1) for x in range(len(face_features))]


if __name__ == '__main__':
	rospy.init_node('live_recognition_node')
	rec = LiveRecognitionNode()

	rate = rospy.Rate(50)
	while not rospy.is_shutdown():
		rec.spin()
