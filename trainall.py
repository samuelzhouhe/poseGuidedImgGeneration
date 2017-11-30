from model_all import Pose_GAN
from dataset_reader import DataLoader
import tensorflow as tf
from config import cfg
import os
import cv2
import datetime
import numpy as np
import scipy.misc

def transform(img):
    img = (img + 1) / 2.0
    return img[:, :, [2, 1, 0]]

dataloader = DataLoader()
model = Pose_GAN()
g1_loss, g2_loss, d_loss= model.build_loss()
tf.summary.scalar("g1loss", g1_loss)
tf.summary.scalar("g2loss", g2_loss)
tf.summary.scalar("dloss", d_loss)

sess = tf.Session()

train_g1 = tf.train.AdamOptimizer(learning_rate=2e-5, beta1=0.5).minimize(g1_loss)
train_g2 = tf.train.AdamOptimizer(learning_rate=1e-6, beta1=0.5).minimize(g2_loss, var_list = model.g2_var)
train_d = tf.train.AdamOptimizer(learning_rate=1e-6, beta1=0.5).minimize(d_loss, var_list = model.d_var)

if not os.path.exists(cfg.LOGDIR):
    os.makedirs(cfg.LOGDIR)
    
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

if (start_itr < cfg.MAXITERATION):
    # step 1: train g1
    for itr in range(start_itr, cfg.MAXITERATION):
        g1_feed, conditional_image, target_image, target_morphologicals = dataloader.next_batch(cfg.BATCH_SIZE, trainorval='TRAIN')
        feed_dict = {model.g1_input: g1_feed, model.ia_input:conditional_image,
                     model.ib_input: target_image, model.mb_plus_1:target_morphologicals}
        sess.run(train_g1, feed_dict=feed_dict)
        if itr %10 == 0:
            train_loss, summaryString = sess.run([g1_loss,summary_merge],feed_dict=feed_dict)
            summary_writer.add_summary(summaryString,itr)
            print("training loss is", train_loss, "itr",itr)

        if itr == cfg.MAXITERATION - 1 or itr%10000==0:
            if itr==cfg.MAXITERATION-1:
                print("Training of G1 done. At iteration ", itr)
            saver.save(sess, cfg.LOGDIR + "/model.ckpt", global_step=itr)

        if itr % 1000 == 0:
            final_output, g2_out, g1_out = sess.run([model.final_output, model.g2_output, model.g1_output],
                                                    feed_dict=feed_dict)

            size = final_output.shape[0]
            dir_name = cfg.RESULT_DIR + '/g1_iter_' + str(itr) + 'at' + str(datetime.datetime.now())
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            for i in range(size):
                name = dir_name + '/sample' + str(i + 1) + 'finalout.jpg'
                scipy.misc.imsave(name, transform(final_output[i]))
                name = dir_name + '/sample' + str(i + 1) + 'g2out.jpg'
                scipy.misc.imsave(name, transform(g2_out[i]))
                name = dir_name + '/sample' + str(i + 1) + 'g1out.jpg'
                scipy.misc.imsave(name, transform(g1_out[i]))
                name_cond = dir_name + '/sample' + str(i + 1) + 'conditionalimg.jpg'
                scipy.misc.imsave(name_cond, transform(conditional_image[i, :, :, :]))
                name_target = dir_name + '/sample' + str(i + 1) + 'target.jpg'
                scipy.misc.imsave(name_target, transform(target_image[i, :, :, :]))

            g1_feed, conditional_image, target_image, target_morphologicals = dataloader.next_batch(cfg.BATCH_SIZE,
                                                                                                    trainorval='VALIDATION')
            feed_dict = {model.g1_input: g1_feed, model.ia_input: conditional_image,
                         model.ib_input: target_image, model.mb_plus_1: target_morphologicals}
            val_g1loss = sess.run(g1_loss,feed_dict=feed_dict)
            print("Validation G1 loss at itr ", itr, " is ", val_g1loss)




saver = tf.train.Saver(max_to_keep=2)
# step 2: train g2 and d
for itr in range(cfg.MAXITERATION-1, 4*cfg.MAXITERATION):
    g1_feed, conditional_image, target_image, target_morphologicals = dataloader.next_batch(cfg.BATCH_SIZE_G2D, trainorval='TRAIN')
    feed_dict = {model.g1_input: g1_feed, model.ia_input:conditional_image,
                 model.ib_input: target_image, model.mb_plus_1:target_morphologicals}
                 
    sess.run([train_g2], feed_dict=feed_dict)

    sess.run([train_d], feed_dict=feed_dict)

    if itr %10 == 0:
        fake_score, real_score, g2loss, dloss, summaryString = sess.run([model.d_fake, model.d_real, g2_loss, d_loss, summary_merge],feed_dict=feed_dict)
        avg_fake_score = np.mean(fake_score)
        avg_real_score = np.mean(real_score)
        summary_writer.add_summary(summaryString,itr)


        print("g2 loss:", g2loss, "|d loss", dloss, "|d real:", avg_real_score, "|d fake", avg_fake_score, "|iteration ", itr)

    if itr == cfg.MAXITERATION - 1 or itr %1000==0:
        saver.save(sess, cfg.LOGDIR + "/model.ckpt", global_step=itr)

    if itr % 100 == 0:
        final_output, g2_out, g1_out = sess.run([model.final_output, model.g2_output, model.g1_output], feed_dict=feed_dict)
        size = final_output.shape[0]

        dir_name = cfg.RESULT_DIR + '/g2_iter_' + str(itr) + 'at' + str(datetime.datetime.now())
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        for i in range(size):
            name = dir_name + '/sample' + str(i + 1) + 'finalout.jpg'
            scipy.misc.imsave(name, transform(final_output[i]))
            name = dir_name + '/sample' + str(i + 1) + 'g2out.jpg'
            scipy.misc.imsave(name, transform(g2_out[i]))
            name = dir_name + '/sample' + str(i + 1) + 'g1out.jpg'
            scipy.misc.imsave(name, transform(g1_out[i]))
            name_cond = dir_name + '/sample' + str(i + 1) + 'conditionalimg.jpg'
            scipy.misc.imsave(name_cond, transform(conditional_image[i, :, :, :]))
            name_target = dir_name + '/sample' + str(i + 1) + 'target.jpg'
            scipy.misc.imsave(name_target, transform(target_image[i, :, :, :]))

        g1_feed, conditional_image, target_image, target_morphologicals = dataloader.next_batch(cfg.BATCH_SIZE,
                                                                                                trainorval='VALIDATION')
        feed_dict = {model.g1_input: g1_feed, model.ia_input: conditional_image,
                     model.ib_input: target_image, model.mb_plus_1: target_morphologicals}
        g2lossvalue, dlossvalue = sess.run([g2_loss, d_loss], feed_dict=feed_dict)
        print("Validation G2 D loss at itr ", itr, " is ", g2lossvalue, " and ", dlossvalue)
        