from model import Pose_GAN
from dataset_reader import DataLoader
import tensorflow as tf
from config import cfg


dataloader = DataLoader()
model = Pose_GAN()
g1_loss, g2_loss, d_loss = model.build_loss()

sess = tf.Session()


train_g1 = tf.train.AdamOptimizer(2e-5).minimize(g1_loss)
train_g2 = tf.train.AdamOptimizer(2e-5).minimize(g2_loss)
train_d = tf.train.AdamOptimizer(2e-5).minimize(d_loss)

saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(cfg.LOGDIR, sess.graph)
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(cfg.LOGDIR)

if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Model restored...")

print("Setting up summary op...")
summary_merge = tf.summary.merge_all()

# step 1: train g1
for itr in range(cfg.MAXITERATION):
    g1_feed, conditional_image, target_image, target_morphologicals = dataloader.next_batch(cfg.BATCH_SIZE)
    feed_dict = {model.g1_input: g1_feed, model.ia_input:conditional_image,
                 model.ib_input:conditional_image, model.mb_plus_1:target_morphologicals}
    sess.run(train_g1, feed_dict=feed_dict)
    if itr %5 == 0:
        train_loss, summaryString = sess.run([g1_loss,summary_merge],feed_dict=feed_dict)
        summary_writer.add_summary(summaryString,itr)
        print("training loss train_loss")

    if itr == cfg.MAXITERATION - 1 or itr == cfg.MAXITERATION // 2:
        saver.save(sess, cfg.BATCH_SIZE + "model.ckpt", itr)

# step 2: train g2 and d
for itr in range(cfg.MAXITERATION, 2*cfg.MAXITERATION):
    g1_feed, conditional_image, target_image, target_morphologicals = dataloader.next_batch(cfg.BATCH_SIZE)
    feed_dict = {model.g1_input: g1_feed, model.ia_input:conditional_image,
                 model.ib_input:conditional_image, model.mb_plus_1:target_morphologicals}
    sess.run([train_g2,train_d], feed_dict=feed_dict)
    if itr == cfg.MAXITERATION - 1 or itr == cfg.MAXITERATION // 2:
        saver.save(sess, cfg.BATCH_SIZE + "model.ckpt", itr)