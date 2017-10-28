from network import Network
from config import cfg
import tensorflow as tf

class Pose_GAN(Network):
	def __init__(self):
		self.inputs = []
		self.g1_input = tf.placeholder(tf.float32, shape = [cfg.BATCH_SIZE] + cfg.G1_INPUT_DATA_SHAPE, name = 'g1_input')
		self.g2_input = tf.placeholder(tf.float32, shape = [cfg.BATCH_SIZE] + cfg.G2_INPUT_DATA_SHAPE, name = 'g2_input')
		self.N = cfg.N
		self.im_width = cfg.INPUT_DATA_SHAPE[1]
		self.im_height = cfg.INPUT_DATA_SHAPE[0]
		self.layers = {'g1_input': self.g1_data, 'g2_input': self.g2_input}
		self.__setup()

	def __setup(self):

		#=============G1 encoder============
		(self.feed('g1_input')
			 .conv2d(3, 128, 1, 1, name = 'conv')
			 .conv2d(3, 128, 1, 1, name = 'block1_conv1')
			 .conv2d(3, 128, 1, 1, name = 'block1_conv2'))
		(self.feed('conv', 'block1_conv2')
			 .add(name = 'add_1')
			 .conv2d(3, 256, 2, 2, name = 'down_sample1', relu = False)
			 .conv2d(3, 256, 1, 1, name = 'block2_conv1')
			 .conv2d(3, 256, 1, 1, name = 'block2_conv2'))
		(self.feed('down_sample1', 'block2_conv2')
			 .add(name = 'add_2')
			 .conv2d(3, 384, 2, 2, name = 'down_sample2', relu = False)
			 .conv2d(3, 384, 1, 1, name = 'block3_conv1',)
			 .conv2d(3, 384, 1, 1, name = 'block3_conv2'))
		(self.feed('down_sample2', 'block3_conv2')
			 .add(name = 'add_3')
			 .conv2d(3, 512, 2, 2, name = 'down_sample3', relu = False)
			 .conv2d(3, 512, 1, 1, name = 'block4_conv1')
			 .conv2d(3, 512, 1, 1, name = 'block4_conv2'))
		(self.feed('down_sample3', 'block4_conv2')
			 .add(name = 'add_4')
			 .conv2d(3, 640, 2, 2, name = 'down_sample4', relu = False)
			 .conv2d(3, 640, 1, 1, name = 'block5_conv1')
			 .conv2d(3, 640, 1, 1, name = 'block5_conv2'))
		(self.feed('down_sample4', 'block5_conv2')
			 .add(name = 'add_5')
			 .conv2d(3, 768, 2, 2, name = 'down_sample5', relu = False)
			 .conv2d(3, 768, 1, 1, name = 'block6_conv1')
			 .conv2d(3, 768, 1, 1, name = 'block6_conv2'))
		(self.feed('down_sample5', 'block6_conv2')
			 .add(name = 'add_6')
			 .fc(64, name = 'fc_1', relu = False))

		W = self.im_width
		H = self.im_height
		for _ in range(self.N - 1):
			W //= 2
			H //= 2
		#=============G1 decoder============
		(self.feed('fc_1').fc(W * H * 768, name = 'fc2', relu = False)
			 .reshape(W, H, 768, name = 'reshape'))
		(self.feed('reshape', 'block6_conv2')
			 .add(name = 'skip_add_1')
			 .conv2d_tran(3, 768, 1, 1, name = 'block1_dconv1')
			 .conv2d_tran(3, 768, 1, 1, name = 'block1_dconv2'))
		(self.feed('skip_add_1', 'block1_dconv2')
			 .add(name = 'back_add_1')
			 .conv2d_tran(3, 640, 2, 2, name = 'up_sample1', relu = False))
		(self.feed('up_sample1', 'block5_conv2')
			 .add(name = 'skip_add_2')
			 .conv2d_tran(3, 640, 1, 1, name = 'block2_dconv1')
			 .conv2d_tran(3, 640, 1, 1, name = 'block2_dconv2'))
		(self.feed('skip_add_2', 'block2_dconv2')
			 .add(name = 'back_add_2')
			 .conv2d_tran(3, 512, 2, 2, name = 'up_sample2', relu = False))
		(self.feed('up_sample2', 'block4_conv2')
			 .add(name = 'skip_add_3')
			 .conv2d_tran(3, 512, 1, 1, name = 'block3_dconv1')
			 .conv2d_tran(3, 512, 1, 1, name = 'block3_dconv2'))
		(self.feed('skip_add_3', 'block3_dconv2')
			 .add(name = 'back_add_3')
			 .conv2d_tran(3, 384, 2, 2, name = 'up_sample3', relu = False))
		(self.feed('up_sample3', 'block3_conv2')
			 .add(name = 'skip_add_4')
			 .conv2d_tran(3, 384, 1, 1, name = 'block4_dconv1')
			 .conv2d_tran(3, 384, 1, 1, name = 'block4_dconv2'))
		(self.feed('skip_add_4', 'block4_dconv2')
			 .add(name = 'back_add_4')
			 .conv2d_tran(3, 256, 2, 2, name = 'up_sample4', relu = False))
		(self.feed('up_sample4', 'block2_conv2')
			 .add(name = 'skip_add_5')
			 .conv2d_tran(3, 256, 1, 1, name = 'block5_dconv1')
			 .conv2d_tran(3, 256, 1, 1, name = 'block5_dconv2'))
		(self.feed('skip_add_5', 'block5_dconv2')
			 .add(name = 'back_add_5')
			 .conv2d_tran(3, 128, 2, 2, name = 'up_sample5', relu = False))
		(self.feed('up_sample5', 'block1_conv2')
			 .add(name = 'skip_add_6')
			 .conv2d_tran(3, 128, 1, 1, name = 'block6_dconv1')
			 .conv2d_tran(3, 128, 1, 1, name = 'block6_dconv2'))
		(self.feed('up_sample5', 'block1_conv2')
			 .add(name = 'back_add_6')
			 .conv2d_tran(3, 3, 1, 1, name = 'g1_result'))

		#=============G1 encoder============
		(self.feed('g2_input', 'g1_result')
			 .concatenate(name = 'concat')
			 .conv2d(3, 128, 1, 1, name = 'conv_g2')
			 .conv2d(3, 128, 1, 1, name = 'g2_block1_conv1')
			 .conv2d(3, 128, 1, 1, name = 'g2_block1_conv2'))
		(self.feed('conv_g2', 'g2_block1_conv2')
			 .add(name = 'g2_add_1')
			 .conv2d(3, 256, 2, 2, name = 'g2_down_sample1', relu = False)
			 .conv2d(3, 256, 1, 1, name = 'g2_block2_conv1')
			 .conv2d(3, 256, 1, 1, name = 'g2_block2_conv2'))
		(self.feed('g2_down_sample1', 'g2_block2_conv2')
			 .add(name = 'g2_add_2')
			 .conv2d(3, 384, 2, 2, name = 'g2_down_sample2', relu = False)
			 .conv2d(3, 384, 1, 1, name = 'g2_block3_conv1')
			 .conv2d(3, 384, 1, 1, name = 'g2_block3_conv2'))
		(self.feed('g2_down_sample2', 'g2_block3_conv2')
			 .add(name = 'g2_add_3')
			 .conv2d(3, 512, 2, 2, name = 'g2_down_sample3', relu = False)
			 .conv2d(3, 512, 1, 1, name = 'g2_block4_conv1')
			 .conv2d(3, 512, 1, 1, name = 'g2_block4_conv2'))
		(self.feed('g2_down_sample3', 'g2_block4_conv2')
			 .add(name = 'g2_add_4'))

		#=============G1 decoder============

	@property
	def output(self):
		return self.layers['g1_result']

	def __build_loss(self):
		pass


if __name__ == '__main__':
	model = G1()















