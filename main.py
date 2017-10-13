import tensorflow as tf

def main(argv=None):
    conditional_img = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name="conditionalImg")
    pose_heatmap = tf.placeholder(tf.float32, shape=[None, 256, 256, 18], name="pose_heatmap")
    g1_input = conditional_img + pose_heatmap
    pass

if __name__ == "__main__":
    tf.app.run()