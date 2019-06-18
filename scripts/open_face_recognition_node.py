#!/usr/bin/env python
import cv2
# import dlib
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
		import openface
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


# wrapper for facenet
# @ref https://github.com/davidsandberg/facenet/blob/master/src/compare.py
# @ref https://github.com/davidsandberg/facenet/blob/master/src/align/detect_face.py
class FaceNetEmbedding:
	def __init__(self):
		import tensorflow as tf
		import align.detect_face
		import facenet

		self.image_size = 160

		package_path = rospkg.RosPack().get_path('open_face_recognition')
		align_model_path = package_path + '/thirdparty/facenet/src/align'

		with tf.Graph().as_default():
			gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
			sess0 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
			with sess0.as_default():
				self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(sess0, align_model_path)

		with tf.Graph().as_default():
			self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
			with self.sess.as_default():
				facenet.load_model(package_path + '/data/facenet')
				self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
				self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
				self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

				# self.feed_dict = {images_placeholder: images, phase_train_placeholder: False}
				# self.emb = sess.run(embeddings, feed_dict=feed_dict)

	def embed(self, bgr_image, is_rgb=False):
		print 'embedding'
		image = bgr_image if not is_rgb else cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
		aligned = self.align(image)

		if aligned is None:
			return None

		images = numpy.expand_dims(aligned, 0)

		feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
		emb = self.sess.run(self.embeddings, feed_dict=feed_dict)

		return emb.flatten()

	def align(self, image):
		import align.detect_face
		from scipy import misc
		import facenet

		minsize = 20                  # minimum size of face
		threshold = [0.6, 0.7, 0.7]   # three steps's threshold
		factor = 0.709                # scale factor
		margin = 44
		bounding_boxes, _ = align.detect_face.detect_face(image, minsize, self.pnet, self.rnet, self.onet, threshold, factor)

		n_faces = bounding_boxes.shape[0]

		if n_faces <= 0:
			print 'failed to detect face...'
			return None

		det = bounding_boxes[:, 0:4]
		det_arr = []
		img_size = numpy.asarray(image.shape)[0:2]

		if n_faces > 1:
			bounding_box_size = (det[:, 2]-det[:, 0])*(det[:, 3]-det[:, 1])
			img_center = img_size / 2
			offsets = numpy.vstack([(det[:, 0]+det[:, 2]) / 2 - img_center[1], (det[:, 1]+det[:, 3]) / 2 - img_center[0]])
			offset_dist_squared = numpy.sum(numpy.power(offsets, 2.0), 0)
			index = numpy.argmax(bounding_box_size - offset_dist_squared * 2.0)   # some extra weight on the centering
			det_arr.append(det[index, :])
		else:
			det_arr.append(numpy.squeeze(det))

		det = numpy.squeeze(det)
		bb = numpy.zeros(4, dtype=numpy.int32)
		bb = numpy.zeros(4, dtype=numpy.int32)
		bb[0] = numpy.maximum(det[0]-margin/2, 0)
		bb[1] = numpy.maximum(det[1]-margin/2, 0)
		bb[2] = numpy.minimum(det[2]+margin/2, img_size[1])
		bb[3] = numpy.minimum(det[3]+margin/2, img_size[0])
		cropped = image[bb[1]: bb[3], bb[0]: bb[2], :]

		aligned = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
		prewhitened = facenet.prewhiten(aligned)

		return prewhitened


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

		embedding_framework = rospy.get_param('~embedding_framework', 'facenet')
		if embedding_framework == 'openface':
			self.embedding = OpenFaceEmbedding(self.image_dim, self.shape_predictor_path, self.network_path)
		elif embedding_framework == 'facenet':
			self.embedding = FaceNetEmbedding()

		print 'ready'

	# getting parameters from rosparam
	# TODO: use dynamic reconfigure
	def get_parameters(self):
		package_path = rospkg.RosPack().get_path('open_face_recognition')
		self.image_dim = rospy.get_param('image_dim', 250)
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
	rec = OpenFaceRecognition()
	rospy.spin()
