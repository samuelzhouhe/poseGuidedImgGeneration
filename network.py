import tensorflow as tf
import numpy as np
import os
import random
import cv2
from config import cfg


WEIGHT_DECAY = cfg.WEIGHT_DECAY

def decorated_layer(layer):
	def wrapper(self, *args,  **kwargs):
		name = kwargs.setdefault('name', self.get_unique_name(layer.__name__))
		
		if len(self.inputs) == 1:
			ipt = self.inputs[0]
		else:
			ipt = self.inputs

		output = layer(self, ipt, *args, **kwargs)
		self.layers[name] = output

		return self.feed(output)

	return wrapper

class Network(object):
	def __init__(self, dataset = None, trainable = True): #data_shape = (height, width)

		self.__trainable = trainable
		if self.__trainable == True:
			assert not dataset == None
			self.__dataset = dataset

		self.__data_shape = dataset.shape
		self.__num_cls = dataset.num_cls
		self.__input_layer = tf.placeholder(tf.float32, shape = [None, self.__data_shape[0], self.__data_shape[1], self.__data_shape[2]], name = 'input')
		self.__image_info = tf.placeholder(tf.float32, shape = [None, self.__num_cls], name = "label")
		self.layers = {'input' : self.__input_layer, 'label' : self.__image_info}
		self.inputs = []
		
		self.setup()

	def feed(self,*args):
		assert len(args) != 0
		self.inputs = []

		for ipt in args:
			if isinstance(ipt, str):
				try:
					ipt = self.layers[ipt]
					print(ipt)
				except KeyError:
					print('Existing layers:',self.layers.keys())
					raise KeyError('Unknown layers %s' %ipt)
			else:
				print(ipt)
			self.inputs.append(ipt)
			
		return self

	def l2_regularizer(self, weight_decay = WEIGHT_DECAY, scope = None):
		def regularizer(tensor):
			with tf.name_scope(scope, default_name='l2_regularizer',values=[tensor]):
				l2_weight = tf.convert_to_tensor(weight_decay, dtype = tensor.dtype.base_dtype, name='weight_decay')
				return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
		return regularizer

	def weight_variable(self, shape, variable_name, scope, collection, reuse, trainable):
		if trainable:
			regularizer = self.l2_regularizer(scope = scope)
		else:
			regularizer = None

		with tf.variable_scope('weight', reuse = reuse):
			var = tf.get_variable(scope + '/' + variable_name, shape = shape, trainable = trainable, collections = collection, regularizer = regularizer)
			tf.summary.histogram(scope + '/weight', var)
			return var

	def bias_variable(self, shape, variable_name, scope, collection, reuse, trainable):
		if trainable:
			regularizer = self.l2_regularizer(scope = scope)
		else:
			regularizer = None

		with tf.variable_scope('bias', reuse = reuse):
			var = tf.get_variable(scope + '/' + variable_name, trainable = trainable, shape = shape, collections = collection, regularizer = regularizer)
			tf.summary.histogram(scope + '/bias', var)
			return var

	def __append(self, appendList, variables):
		if not appendList is None:
			assert isinstance(appendList, list)
			assert isinstance(variables, list)
			assert len(variables) > 0
			for item in variables:
				appendList.append(item)


	@decorated_layer
	def  	conv2d(self, input_data, k_w, k_d, s_w, s_h, name, collection = None, 
				scope = None, relu = True, padding = 'SAME', appendList = None, reuse = False, trainable = True):

		depth = input_data.get_shape().as_list()[-1]

		if reuse == True:
			assert not scope is None

		if scope is None:
			scope = name

		#kernel
		with tf.name_scope(scope):
			kernel = self.weight_variable([k_w, k_w, depth, k_d], 'kernel', scope, collection, reuse = reuse, trainable = trainable)
			bias = self.bias_variable([k_d], 'b', scope, collection, reuse = reuse, trainable = trainable)
			conv2d = tf.nn.conv2d(input_data, kernel, strides = [1, s_h, s_w, 1], padding = padding)

			self.__append(appendList, [kernel, bias])

			if relu:
				return tf.nn.relu(tf.nn.bias_add(conv2d, bias))
			else:
				return tf.nn.bias_add(conv2d, bias)

	@decorated_layer
	def reshape(self, input_data, *data_shape, name = None):
		assert len(data_shape) > 0
		return tf.reshape(input_data, data_shape)

	@decorated_layer
	def conv2d_tran(self, input_data, k_w, k_d, s_w, s_h, name, output_shape = None, 
			scope = None, collection = None, relu = True, padding = 'SAME', appendList = None, reuse = False, trainable = True):

		depth = input_data.get_shape().as_list()[-1]

		if reuse == True:
			assert not scope is None

		if scope is None:
			scope = name

		if output_shape is None:
			output_shape = input_data.get_shape().as_list()
			output_shape[0] = cfg.BATCH_SIZE
			output_shape[1] *= s_h
			output_shape[2] *= s_w
			output_shape[3] = k_d


		#kernel
		with tf.name_scope(scope):
			kernel = self.weight_variable([k_w, k_w, k_d, depth], 'kernel', scope, collection, reuse = reuse, trainable = trainable)
			bias = self.bias_variable([k_d], 'b', scope, collection, reuse = reuse, trainable = trainable)
			conv2d = tf.nn.conv2d_transpose(input_data, kernel, output_shape, strides = [1, s_h, s_w, 1], padding = padding)

			self.__append(appendList, [kernel, bias])

			if relu:
				return tf.nn.relu(tf.nn.bias_add(conv2d, bias))
			else:
				return tf.nn.bias_add(conv2d, bias)

	@decorated_layer
	def max_pooling(self, input_data, name, padding = 'SAME'):
		return tf.nn.max_pool(input_data, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],padding = padding, name = name)


	@decorated_layer
	def fc(self, input_data, output_dim, name, collection = None, scope = None, relu = True, appendList = None, reuse = False, trainable = True):
		assert not isinstance(input_data, list)

		if reuse == True:
			assert not scope is None

		if scope is None:
			scope = name

		shape = input_data.get_shape().as_list()
		
		size = 1

		for i in shape[1:]:
			size *= i

		with tf.name_scope(scope):
			if len(shape) == 4:
				input_data = tf.reshape(input_data, [-1, size])
		
			w = self.weight_variable([size, output_dim], 'w', scope, collection, reuse = reuse, trainable = trainable)
			b = self.bias_variable([output_dim], 'b', scope, collection, reuse = reuse, trainable = trainable)

			self.__append(appendList, [w, b])

			if relu:
				op = tf.nn.relu_layer
			else:
				op = tf.nn.xw_plus_b

			return op(input_data, w, b)


	@decorated_layer
	def soft_max(self, input_data, name, labels = None, loss = True):
		#print(input_data.get_shape().as_list())
		if loss:
			if labels is None:
				labels = self.layers['label']
			return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = input_data), name = name)
		else:
			return tf.nn.softmax(input_data, name = name)

	@decorated_layer
	def sigmoid(self, input_data, name, labels = None, loss = True):
		if loss:
			if labels is None:
				labels = self.layers['label']
			return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = input_data, labels = labels), name = name)
		else:
			return tf.nn.sigmoid(input_data, name = name)
		

	@decorated_layer
	def drop_out(self, input_data, name, keep_prob_name):
		kp = tf.placeholder(tf.float32, name = keep_prob_name)
		self.layers[keep_prob_name] = kp
		return tf.nn.dropout(input_data, keep_prob = kp, name = name)

	@decorated_layer
	def producer(self, input_data, output_queue, name):
		output_queue.put(input_data)
		return input_data

	@decorated_layer
	def add(self, input_data, name):
		assert len(input_data) == 2
		return tf.add(input_data[0], input_data[1], name = name)

	@decorated_layer
	def concatenate(self, input_data, name, axis = 1):
		assert len(input_data) >= 2
		return tf.concat(input_data, axis = axis)

	@decorated_layer
	def leaky_relu(self, input_data, name, alpha = 0.2):
		return tf.nn.relu(input_data) - alpha * tf.nn.relu(-input_data)

	@decorated_layer
	def weight_sum(self, input_data, name, collection, stddev = 0.1):
		shape = input_data[0].get_shape().as_list()
		for data in input_data:
			assert data.get_shape().as_list() == shape

		weight_variables = [tf.multiply(weight_variable(shape, collection, stddev = stddev), input_data[i]) for i in len(input_data)]
		return tf.reduce_sum(weight_variables, axis = 1)

	@decorated_layer

	def batch_normalization(self, input_data, name, scope = None, relu = True, decay = 0.9, epsilon = 1e-5, updates_collections = tf.GraphKeys.UPDATE_OPS, trainable = False, appendList = None, reuse = False):
		with tf.variable_scope(scope, reuse = reuse):
			if reuse == True:
				assert not scope is None
			temp_layer =  tf.contrib.layers.batch_norm(input_data, decay = decay, scale = True, 
													center=True, variables_collections = scope, epsilon = epsilon, is_training = trainable)
		if relu:
			return tf.nn.relu(temp_layer)
		else:
			return temp_layer

	@decorated_layer
	def stop_gradient(self, input_data, name):
		return tf.stop_gradient(input_data, name = name)

	@decorated_layer
	def tanh(self, input_data, name):
		return tf.tanh(input_data)

	def setup(self):
		raise NotImplementedError('Function setup(self) must be implemented!')


	def get_unique_name(self, layer_name):
		count = len([name for name, _ in self.layers.items() if name.startswith(layer_name)])
		new_name = layer_name + '_' + str(count + 1)
		return new_name

