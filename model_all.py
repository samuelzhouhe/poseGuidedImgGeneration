from network import Network
from config import cfg
import tensorflow as tf

class Pose_GAN(Network):
	def __init__(self, traing1ornot=True):
		self.inputs = []
		self.g1_input = tf.placeholder(tf.float32, shape = [cfg.BATCH_SIZE] + cfg.G1_INPUT_DATA_SHAPE, name = 'g1_input')
		self.ia_input = tf.placeholder(tf.float32, shape = [cfg.BATCH_SIZE] + cfg.IMAGE_SHAPE, name = 'ia_input')
		self.ib_input = tf.placeholder(tf.float32, shape = [cfg.BATCH_SIZE] + cfg.IMAGE_SHAPE, name = 'ib_input')
		self.traing1ornot = traing1ornot
		self.N = cfg.N
		self.im_width = cfg.G1_INPUT_DATA_SHAPE[1]
		self.im_height = cfg.G1_INPUT_DATA_SHAPE[0]
		self.g2_var = []
		self.d_var = []
		self.layers = {'g1_input': self.g1_input, 'ia_input': self.ia_input, 'ib_input': self.ib_input}
		self.__setup()

	def __setup(self):

		#=============G1 encoder============
		print('G1 encoder')
		with tf.name_scope('G1'):
			(self.feed('g1_input')
				 .conv2d(3, 128, 1, 1, name = 'conv', trainable = self.traing1ornot)
				 .conv2d(3, 128, 1, 1, name = 'block1_conv1', trainable = self.traing1ornot)
				 .conv2d(3, 128, 1, 1, name = 'block1_conv2', trainable = self.traing1ornot))
			(self.feed('conv', 'block1_conv2')
				 .add(name = 'add_1')
				 .conv2d(3, 256, 2, 2, name = 'down_sample1', relu = False, trainable = self.traing1ornot)
				 .conv2d(3, 256, 1, 1, name = 'block2_conv1', trainable = self.traing1ornot)
				 .conv2d(3, 256, 1, 1, name = 'block2_conv2', trainable = self.traing1ornot))
			(self.feed('down_sample1', 'block2_conv2')
				 .add(name = 'add_2')
				 .conv2d(3, 384, 2, 2, name = 'down_sample2', relu = False, trainable = self.traing1ornot)
				 .conv2d(3, 384, 1, 1, name = 'block3_conv1', trainable = self.traing1ornot)
				 .conv2d(3, 384, 1, 1, name = 'block3_conv2', trainable = self.traing1ornot))
			(self.feed('down_sample2', 'block3_conv2')
				 .add(name = 'add_3')
				 .conv2d(3, 512, 2, 2, name = 'down_sample3', relu = False, trainable = self.traing1ornot)
				 .conv2d(3, 512, 1, 1, name = 'block4_conv1', trainable = self.traing1ornot)
				 .conv2d(3, 512, 1, 1, name = 'block4_conv2', trainable = self.traing1ornot))
			(self.feed('down_sample3', 'block4_conv2')
				 .add(name = 'add_4')
				 .conv2d(3, 640, 2, 2, name = 'down_sample4', relu = False, trainable = self.traing1ornot)
				 .conv2d(3, 640, 1, 1, name = 'block5_conv1', trainable = self.traing1ornot)
				 .conv2d(3, 640, 1, 1, name = 'block5_conv2', trainable = self.traing1ornot))
			(self.feed('down_sample4', 'block5_conv2')
				 .add(name = 'add_5')
				 .conv2d(3, 768, 2, 2, name = 'down_sample5', relu = False, trainable = self.traing1ornot)
				 .conv2d(3, 768, 1, 1, name = 'block6_conv1', trainable = self.traing1ornot)
				 .conv2d(3, 768, 1, 1, name = 'block6_conv2', trainable = self.traing1ornot))
			(self.feed('down_sample5', 'block6_conv2')
				 .add(name = 'add_6')
				 .fc(64, name = 'fc_1', relu = False, trainable = self.traing1ornot))

			W = self.im_width
			H = self.im_height
			for _ in range(self.N - 1):
				W //= 2
				H //= 2

		#=============G1 decoder============
			print('=============G1 decoder=============')
			(self.feed('fc_1').fc(W * H * 768, name = 'fc2', relu = False, trainable = self.traing1ornot)
				 .reshape(cfg.BATCH_SIZE, W, H, 768, name = 'reshape'))
			(self.feed('reshape', 'block6_conv2')
				 .add(name = 'skip_add_1')
				 .conv2d_tran(3, 768, 1, 1, name = 'block1_dconv1', trainable = self.traing1ornot)
				 .conv2d_tran(3, 768, 1, 1, name = 'block1_dconv2', trainable = self.traing1ornot))
			(self.feed('skip_add_1', 'block1_dconv2')
				 .add(name = 'back_add_1')
				 .conv2d_tran(3, 640, 2, 2, name = 'up_sample1', relu = False, trainable = self.traing1ornot))
			(self.feed('up_sample1', 'block5_conv2')
				 .add(name = 'skip_add_2')
				 .conv2d_tran(3, 640, 1, 1, name = 'block2_dconv1', trainable = self.traing1ornot)
				 .conv2d_tran(3, 640, 1, 1, name = 'block2_dconv2', trainable = self.traing1ornot))
			(self.feed('skip_add_2', 'block2_dconv2')
				 .add(name = 'back_add_2')
				 .conv2d_tran(3, 512, 2, 2, name = 'up_sample2', relu = False, trainable = self.traing1ornot))
			(self.feed('up_sample2', 'block4_conv2')
				 .add(name = 'skip_add_3')
				 .conv2d_tran(3, 512, 1, 1, name = 'block3_dconv1', trainable = self.traing1ornot)
				 .conv2d_tran(3, 512, 1, 1, name = 'block3_dconv2', trainable = self.traing1ornot))
			(self.feed('skip_add_3', 'block3_dconv2')
				 .add(name = 'back_add_3')
				 .conv2d_tran(3, 384, 2, 2, name = 'up_sample3', relu = False, trainable = self.traing1ornot))
			(self.feed('up_sample3', 'block3_conv2')
				 .add(name = 'skip_add_4')
				 .conv2d_tran(3, 384, 1, 1, name = 'block4_dconv1', trainable = self.traing1ornot)
				 .conv2d_tran(3, 384, 1, 1, name = 'block4_dconv2', trainable = self.traing1ornot))
			(self.feed('skip_add_4', 'block4_dconv2')
				 .add(name = 'back_add_4')
				 .conv2d_tran(3, 256, 2, 2, name = 'up_sample4', relu = False, trainable = self.traing1ornot))
			(self.feed('up_sample4', 'block2_conv2')
				 .add(name = 'skip_add_5')
				 .conv2d_tran(3, 256, 1, 1, name = 'block5_dconv1', trainable = self.traing1ornot)
				 .conv2d_tran(3, 256, 1, 1, name = 'block5_dconv2', trainable = self.traing1ornot))
			(self.feed('skip_add_5', 'block5_dconv2')
				 .add(name = 'back_add_5')
				 .conv2d_tran(3, 128, 2, 2, name = 'up_sample5', relu = False, trainable = self.traing1ornot))
			(self.feed('up_sample5', 'block1_conv2')
				 .add(name = 'skip_add_6')
				 .conv2d_tran(3, 128, 1, 1, name = 'block6_dconv1', trainable = self.traing1ornot)
				 .conv2d_tran(3, 128, 1, 1, name = 'block6_dconv2', trainable = self.traing1ornot))
			(self.feed('up_sample5', 'block1_conv2')
				 .add(name = 'back_add_6')
				 .conv2d_tran(3, 3, 1, 1, name = 'dconv_out', relu = False, trainable = self.traing1ornot)
				 .tanh(name = 'g1_result'))

			#=============G2 encoder============
		with tf.name_scope('G2'):
			print('=============G2 encoder=============')
			(self.feed('g1_result').stop_gradient(name = 'barrier'))
			(self.feed('ia_input', 'barrier')
				 .concatenate(name = 'concat', axis = -1)
				 .conv2d(3, 128, 1, 1, name = 'conv_g2', appendList = self.g2_var)
				 .conv2d(3, 128, 1, 1, name = 'g2_block1_conv1', appendList = self.g2_var)
				 .conv2d(3, 128, 1, 1, name = 'g2_block1_conv2', appendList = self.g2_var))
			(self.feed('conv_g2', 'g2_block1_conv2')
				 .add(name = 'g2_add_1')
				 .conv2d(3, 256, 2, 2, name = 'g2_down_sample1', relu = False, appendList = self.g2_var)
				 .conv2d(3, 256, 1, 1, name = 'g2_block2_conv1', appendList = self.g2_var)
				 .conv2d(3, 256, 1, 1, name = 'g2_block2_conv2', appendList = self.g2_var))
			(self.feed('g2_down_sample1', 'g2_block2_conv2')
				 .add(name = 'g2_add_2')
				 .conv2d(3, 384, 2, 2, name = 'g2_down_sample2', relu = False, appendList = self.g2_var)
				 .conv2d(3, 384, 1, 1, name = 'g2_block3_conv1', appendList = self.g2_var)
				 .conv2d(3, 384, 1, 1, name = 'g2_block3_conv2', appendList = self.g2_var))
			(self.feed('g2_down_sample2', 'g2_block3_conv2')
				 .add(name = 'g2_add_3')
				 .conv2d(3, 512, 2, 2, name = 'g2_down_sample3', relu = False, appendList = self.g2_var)
				 .conv2d(3, 512, 1, 1, name = 'g2_middle_conv1', appendList = self.g2_var)
				 .conv2d(3, 512, 1, 1, name = 'g2_middle_conv2', appendList = self.g2_var))


			#=============G2 decoder============
			print('=============G2 decoder=============')
			(self.feed('g2_down_sample3', 'g2_middle_conv2')
				 .add(name = 'g2_add_4')
				 .conv2d_tran(3, 384, 2, 2, name = 'g2_up_sample1', relu = False, appendList = self.g2_var))
			(self.feed('g2_up_sample1', 'g2_block3_conv2')
				 .add(name = 'g2_skip_add1')
				 .conv2d_tran(3, 384, 1, 1, name = 'g2_block1_dconv1', appendList = self.g2_var)
				 .conv2d_tran(3, 384, 1, 1, name = 'g2_block1_dconv2', appendList = self.g2_var))
			(self.feed('g2_skip_add1', 'g2_block1_dconv2')
				 .add(name = 'g2_back_add1')
				 .conv2d_tran(3, 256, 2, 2, name = 'g2_up_sample2', relu = False, appendList = self.g2_var))
			(self.feed('g2_up_sample2', 'g2_block2_conv2')
				 .add(name = 'g2_skip_add2')
				 .conv2d_tran(3, 256, 1, 1, name = 'g2_block2_dconv1', appendList = self.g2_var)
				 .conv2d_tran(3, 256, 1, 1, name = 'g2_block2_dconv2', appendList = self.g2_var))
			(self.feed('g2_skip_add2', 'g2_block2_dconv2')
				 .add(name = 'g2_back_add2')
				 .conv2d_tran(3, 128, 2, 2, name = 'g2_up_sample3', relu = False, appendList = self.g2_var))
			(self.feed('g2_up_sample3', 'g2_block1_conv2')
				 .add(name = 'g2_skip_add3')
				 .conv2d_tran(3, 128, 1, 1, name = 'g2_block3_dconv1', appendList = self.g2_var)
				 .conv2d_tran(3, 128, 1, 1, name = 'g2_block3_dconv2', appendList = self.g2_var))
			(self.feed('g2_skip_add3', 'g2_block3_dconv2')
				 .add(name = 'g2_back_add3')
				 .conv2d_tran(3, 3, 1, 1, name = 'g2_dconv_out', relu = False, appendList = self.g2_var)
				 .tanh(name = 'g2_result'))

		#=============Final output============
		print('=============Final output=============')
		(self.feed('barrier', 'g2_result')
			 .add(name = 'final_output'))

		#=============Discriminator============
		print('=============Discriminator=============')
		with tf.variable_scope('Discriminator'):
			(self.feed('ia_input', 'ib_input')
				 .concatenate(name = 'd_real_input', axis = -1)
				 .conv2d(5, 64, 2, 2, name = 'd_real_conv1', scope = 'd_conv_1', relu = False)
				 .leaky_relu(name = 'd_real_lrelu1')
				 .conv2d(5, 128, 2, 2, name = 'd_real_conv2', scope = 'd_conv_2', relu = False)
				 .batch_normalization(name = 'd_real_bn1', scope = 'd_bn1',relu = False, trainable = True, updates_collections = None)
				 .leaky_relu(name = 'd_real_lrelu2')
				 .conv2d(5, 256, 2, 2, name = 'd_real_conv3', scope = 'd_conv_3', relu = False)
				 .batch_normalization(name = 'd_real_bn2', scope = 'd_bn2', relu = False, trainable = True, updates_collections = None)
				 .leaky_relu(name = 'd_real_lrelu3')
				 .conv2d(5, 512, 2, 2, name = 'd_real_conv4', scope = 'd_conv_4', relu = False)
				 .batch_normalization(name = 'd_real_bn3', scope = 'd_bn3', relu = False, trainable = True, updates_collections = None)
				 .leaky_relu(name = 'd_real_lrelu4')
				 .fc(1, name = 'logit_real', scope = 'logit', relu = False)
				 .sigmoid(name = 'd_real', loss = False))

			(self.feed('ia_input', 'final_output')
				 .concatenate(name = 'd_fake_input', axis = -1)
				 .conv2d(5, 64, 2, 2, name = 'd_fake_conv1', scope = 'd_conv_1', relu = False, reuse = True)
				 .leaky_relu(name = 'd_fake_lrelu1')
				 .conv2d(5, 128, 2, 2, name = 'd_fake_conv2', scope = 'd_conv_2', relu = False, reuse = True)
				 .batch_normalization(name = 'd_fake_bn1', scope = 'd_bn1',relu = False, trainable = True, updates_collections = None, reuse = True)
				 .leaky_relu(name = 'd_fake_lrelu2')
				 .conv2d(5, 256, 2, 2, name = 'd_fake_conv3', scope = 'd_conv_3', relu = False, reuse = True)
				 .batch_normalization(name = 'd_fake_bn2', scope = 'd_bn2', relu = False, trainable = True, updates_collections = None, reuse = True)
				 .leaky_relu(name = 'd_fake_lrelu3')
				 .conv2d(5, 512, 2, 2, name = 'd_fake_conv4', scope = 'd_conv_4', relu = False, reuse = True)
				 .batch_normalization(name = 'd_fake_bn3', scope = 'd_bn3', relu = False, trainable = True, updates_collections = None, reuse = True)
				 .leaky_relu(name = 'd_fake_lrelu4')
				 .fc(1, name = 'logit_fake', scope = 'logit', relu = False, reuse = True)
				 .sigmoid(name = 'd_fake', loss = False))


			t_var = tf.trainable_variables()
			self.d_var = [var for var in t_var if 'd_' in var.name]

	@property
	def d_fake(self):
		return self.layers['d_fake']

	@property
	def d_real(self):
		return self.layers['d_real']

	@property
	def g1_output(self):
		return self.layers['g1_result']

	@property
	def g2_output(self):
		return self.layers['g2_result']

	@property
	def final_output(self):
		return self.layers['final_output']

	@property
	def mb_plus_1(self):
		return self.layers['mb_plus_1']

	def build_loss(self):
		#=============g1 loss============
		self.layers['mb_plus_1'] = tf.placeholder(tf.float32, shape = [cfg.BATCH_SIZE] + cfg.IMAGE_SHAPE[:2] + [1], name = 'mb_plus_1')
		l1_distance = tf.abs(tf.multiply(self.layers['g1_result'] - self.layers['ib_input'], self.layers['mb_plus_1']))

		self.layers['g1_loss'] = tf.reduce_mean(tf.reduce_sum(l1_distance, axis = [1, 2, 3]))

		#=============discriminator loss============
		(self.feed('logit_real')
			 .sigmoid(name = 'real_loss', labels = tf.ones_like(self.layers['logit_real']), loss = True))
		(self.feed('logit_fake')
			 .sigmoid(name = 'fake_loss', labels = tf.zeros_like(self.layers['logit_fake']), loss = True))
		self.layers['d_loss'] = tf.reduce_mean(self.layers['fake_loss'] + self.layers['real_loss'])

		#=============g2 loss============
		# (self.feed('logit_fake')
		# 	 .sigmoid(name = 'g2_adv_loss', labels = tf.ones_like(self.layers['logit_fake']), loss = True))
		likely_hood = -tf.reduce_mean(tf.log(self.layers['d_fake']))

		l1_distance2 = tf.reduce_sum(tf.abs(tf.multiply(self.layers['final_output'] - self.layers['ib_input'], self.layers['mb_plus_1'])), axis = [1, 2, 3])
		#self.layers['g2_loss'] = tf.reduce_mean(self.layers['g2_adv_loss']) + cfg.LAMBDA * tf.reduce_mean(l1_distance2)
		self.layers['g2_loss'] = likely_hood + cfg.LAMBDA * tf.reduce_mean(l1_distance2)
		#=============l2 regularization loss============
		# self.layers['l2_reg_loss'] = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

		return self.layers['g1_loss'], self.layers['g2_loss'], self.layers['d_loss']

if __name__ == '__main__':
	model = Pose_GAN()
	a, b, c = model.build_loss()
	print('==========================')
	counter = 0
	for var in model.g2_var:
		counter += 1
		print(var.name)
	print('number of variables:', counter)
	print('==========================')
	for var in model.d_var:
		print(var.name)















