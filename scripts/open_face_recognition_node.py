#!/usr/bin/env python
import cv2
import dlib
import numpy
import rospy
import rospkg
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from cv_bridge import CvBridge
from open_face_recognition.srv import *


# wrapper for openface
class OpenFaceEmbedding:
	# constructor
	def __init__(self, image_dim, shape_predictor_path, network_path):
		import openface
		print 'network_path', network_path
		print 'loading network...',
		self.image_dim = image_dim
		self.align = openface.AlignDlib(shape_predictor_path)
		self.net = openface.TorchNeuralNet(network_path, self.image_dim)
		print 'done'

	# this method calculates a real vector which represents the given face image
	def embed(self, bgr_image, is_rgb=False):

		rgb_image = bgr_image
		if not is_rgb:
			rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

		bb = self.align.getLargestFaceBoundingBox(rgb_image)
		# bb = self.align_hueristic(rgb_image)
		if bb is None:
			if rgb_image.shape[0] > 32:
				scale = 0.5
				print 're-try with smaller size', int(rgb_image.shape[0] * scale)
				rgb_image = cv2.resize(rgb_image, (int(rgb_image.shape[1] * scale), int(rgb_image.shape[0] * scale)))
				return self.embed(rgb_image, True)
			print 'warning : failed to obtain face bounding box!!'
			return None
			# if you want to obtain feature vectors even when the face landmark detection failed,
			# comment out the above line (return None) and use the heuristic alignment
			print 'warning : use heuristic align!!'
			bb = self.align_hueristic(rgb_image)

		aligned = self.align.align(self.image_dim, rgb_image, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
		if aligned is None:
			print 'warning : failed to align input face!!'
			return None

		return self.net.forward(aligned)

	# align a face image hueristically
	# the parameters are determined from facescrub dataset
	def align_hueristic(self, rgb_image):
		top = int(rgb_image.shape[0] * 0.25)
		bottom = int(rgb_image.shape[0] * 0.9)
		left = int(rgb_image.shape[1] * 0.15)
		right = int(rgb_image.shape[1] * 0.85)
		return dlib.rectangle(top, left, bottom, right)


class FaceNetEmbedding:
	def __init__(self):
		import tensorflow as tf

		with tf.Graph().as_default():
			gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
			sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
			with sess.as_default():
				pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

	def embed(self, bgr_image, is_rgb=False):
		pass

# ros node
class OpenFaceRecognition:
	# constructor
	def __init__(self):
		print '--- open_face_recognition---'
		rospy.init_node('open_face_recognition')
		self.comp_service = rospy.Service('compare_images', CompareImages, self.compare_images)
		self.feat_service = rospy.Service('calc_image_features', CalcImageFeatures, self.calc_image_features)

		self.get_parameters()
		self.bridge = CvBridge()
		self.embedding = OpenFaceEmbedding(self.image_dim, self.shape_predictor_path, self.network_path)

	# getting parameters from rosparam
	# TODO: use dynamic reconfigure
	def get_parameters(self):
		package_path = rospkg.RosPack().get_path('open_face_recognition')
		self.image_dim = rospy.get_param('image_dim', 96)
		self.shape_predictor_path = package_path + rospy.get_param('shape_predictor_path', '/data/shape_predictor_68_face_landmarks.dat')
		self.network_path = package_path + rospy.get_param('network_path', '/data/nn4.small2.v1.t7')

	# ReloadModelFile service
	# this services replaces the network model with the given model file
	def reload_model_file(self, req):
		print 'reload', req.model_path
		self.get_parameters()
		self.embedding = OpenFaceEmbedding(self.image_dim, self.shape_predictor_path, req.model_path)
		return True

	# CompareImage service
	# this service receives a pair of face images and then returns the distance between the images
	def compare_images(self, req):
		while not hasattr(self, 'embedding'):
			rospy.sleep(0.1)

		lhs_image = self.bridge.imgmsg_to_cv2(req.lhs)
		rhs_image = self.bridge.imgmsg_to_cv2(req.rhs)
		lhs_image = cv2.cvtColor(lhs_image, cv2.COLOR_BGR2RGB)
		rhs_image = cv2.cvtColor(rhs_image, cv2.COLOR_BGR2RGB)

		lhs_vec = self.embedding.embed(lhs_image)
		rhs_vec = self.embedding.embed(rhs_image)
		return numpy.linalg.norm(lhs_vec - rhs_vec)

	# CalcImageFeatures service
	# this service receives an image and then returns a feature vector exmodelracted from the image
	def calc_image_features(self, req):
		while not hasattr(self, 'embedding'):
			rospy.sleep(0.1)

		image = self.bridge.imgmsg_to_cv2(req.image)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		features = self.embedding.embed(image)

		if features is None:
			features = numpy.array([0])

		res = Float32MultiArray()
		res.layout = MultiArrayLayout()
		res.layout.dim = [MultiArrayDimension('features', features.shape[0], features.shape[0])]
		res.layout.data_offset = 0
		res.data = features.tolist()

		return res

if __name__ == '__main__':
	# rec = OpenFaceRecognition()
	# rospy.spin()

	embed = FaceNetEmbedding()
	face_img = cv2.imread('/tmp/test.jpg')

	embed.embed(face_img)
