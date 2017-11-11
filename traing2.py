from model_all import Pose_GAN
from dataset_reader import DataLoader
import tensorflow as tf
from config import cfg
import os
import cv2
import datetime


dataloader = DataLoader()
model = Pose_GAN(traing1ornot=False)
g2_loss, d_loss, _ = model.build_loss()
tf.summary.scalar("g2loss", g2_loss)
tf.summary.scalar("dloss", d_loss)

sess = tf.Session()
optimizer = tf.train.AdamOptimizer(learning_rate=2e-5, beta1=0.5)
train_g2 = optimizer.minimize(g2_loss)
train_d = optimizer.minimize(d_loss)

saver = tf.train.Saver(max_to_keep=2)
summary_writer = tf.summary.FileWriter(cfg.LOGDIR, sess.graph)
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(cfg.LOGDIR)

start_itr = 0
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Model restored...")
    start_itr = int(ckpt.model_checkpoint_path.split('-')[1])
    print("starting from iteration", start_itr)

print("Setting up summary op...")
summary_merge = tf.summary.merge_all()

if not os.path.exists(cfg.RESULT_DIR):
    os.makedirs(cfg.RESULT_DIR)

# step 2: train g2 and d
if start_itr >= cfg.MAXITERATION and start_itr < 2*cfg.MAXITERATION:
    for itr in range(start_itr, 2*cfg.MAXITERATION):
        g1_feed, conditional_image, target_image, target_morphologicals = dataloader.next_batch(cfg.BATCH_SIZE, trainorval='TRAIN')
        feed_dict = {model.g1_input: g1_feed, model.ia_input:conditional_image,
                     model.ib_input: target_image, model.mb_plus_1:target_morphologicals}
        sess.run([train_g2,train_d], feed_dict=feed_dict)

        if itr %10 == 0:
            g2loss, dloss, summaryString = sess.run([g2_loss, d_loss, summary_merge],feed_dict=feed_dict)
            summary_writer.add_summary(summaryString,itr)
            print("training loss when training g and d is", g2loss, "and",dloss, "at iteration ", itr)

        if itr == cfg.MAXITERATION - 1 or itr %50==0:
            saver.save(sess, cfg.LOGDIR + "/model.ckpt", global_step=itr)

        if itr % 50 == 0:
            final_output, g2_out, g1_out = sess.run([model.final_output, model.g2_output, model.g1_output], feed_dict=feed_dict)
            size = final_output.shape[0]
            dir_name = cfg.RESULT_DIR + '/g2_iter_' + str(itr) + 'at' + str(datetime.datetime.now())
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            for i in range(size):
                name = dir_name + '/sample' + str(i + 1) + 'finalout.jpg'
                cv2.imwrite(name, final_output[i])
                name = dir_name + '/sample' + str(i + 1) + 'g2out.jpg'
                cv2.imwrite(name, g2_out[i])
                name = dir_name + '/sample' + str(i + 1) + 'g1out.jpg'
                cv2.imwrite(name, g1_out[i])
                name_cond = dir_name + '/sample' + str(i + 1) + 'conditionalimg.jpg'
                cv2.imwrite(name_cond, conditional_image[i, :, :, :])
                name_target = dir_name + '/sample' + str(i + 1) + 'target.jpg'
                cv2.imwrite(name_target, target_image[i, :, :, :])

            g1_feed, conditional_image, target_image, target_morphologicals = dataloader.next_batch(cfg.BATCH_SIZE,
                                                                                                    trainorval='VALIDATION')
            feed_dict = {model.g1_input: g1_feed, model.ia_input: conditional_image,
                         model.ib_input: target_image, model.mb_plus_1: target_morphologicals}
            g2lossvalue, dlossvalue = sess.run([g2_loss, d_loss], feed_dict=feed_dict)
            print("Validation G2 D loss at itr ", itr, " is ", g2lossvalue, " and ", dlossvalue)