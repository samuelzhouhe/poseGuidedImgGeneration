from model import Pose_GAN
from dataset_reader import DataLoader
import tensorflow as tf
from config import cfg


dataloader = DataLoader()
model = Pose_GAN()
g1_loss, g2_loss, d_loss = model.build_loss()

sess = tf.Session()


train_g1 = tf.train.AdamOptimizer(1e-3).minimize(g1_loss)

saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(cfg.LOGDIR, sess.graph)
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(cfg.LOGDIR)

if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Model restored...")


for _ in cfg.MAXITERATION:
    g1_feed, conditional_image, target_image, target_morphologicals = dataloader.next_batch(cfg.BATCH_SIZE)
    feed_dict = {model.g1_input: g1_feed, model.g2_input:conditional_image,
                 model.da_input:conditional_image, model.db_input:target_image}