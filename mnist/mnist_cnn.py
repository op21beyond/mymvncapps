# Lab 13 Saver and Restore
import tensorflow as tf
import random
# import matplotlib.pyplot as plt
import numpy as np
import os

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnistbeginners for
# more information about the mnist dataset

# parameters
learning_rate = 0.001
#learning_rate = 0.01
training_epochs = 15
#training_epochs = 5
batch_size = 100
use_batch_normalization = False #
use_dropout = False

is_training = False

enable_summary = False

n_classes = 10

CHECK_POINT_DIR = TB_SUMMARY_DIR = './tb/mnist_cnn'

# input place holders
if is_training:
   X = tf.placeholder(tf.float32, [None, 28,28,1], name='input')
   Y = tf.placeholder(tf.float32, [None, 10], name='label')
else:
   X = tf.placeholder(tf.float32, [1, 28,28,1], name='input')

# Image input
if enable_summary:
   tf.summary.image('in', X, 3)

# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
if use_dropout:
   keep_prob = tf.placeholder(tf.float32)

# weights & bias for nn layers
# http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
arg_scope = tf.contrib.framework.arg_scope
batch_normalization = tf.contrib.layers.batch_norm
w_initializer = tf.contrib.layers.xavier_initializer()
#w_initializer = tf.contrib.layers.variance_scaling_initializer()
w_regularizer = tf.contrib.layers.l2_regularizer(0.05)

n_clayers = 3 # number of convolution layers
C  = [1, 32, 64, 128] # channels
CK = [3, 3,  3,  3] # kernel height, width
CS = [1, 1,  1,  1] # conv2d strides
PK = [2, 2,  2,  2] # pooling kernel size
PS = [2, 2,  2,  2] # pooling strides

with arg_scope([batch_normalization], 
                 decay=0.99, center=True, scale=True):

   if enable_summary:
      tf.summary.histogram("X", X)

   L = X

   for layer in range(n_clayers):
      with tf.variable_scope('conv{}'.format(layer)):
          W = tf.get_variable(name="W", 
                   shape=[CK[layer],CK[layer],C[layer],C[layer+1]],
                   regularizer=w_regularizer,
                   initializer=w_initializer,
                   )

          # 2D convolution
          L = tf.nn.conv2d(
             input = L,
             filter = W,
             strides = [1,CS[layer],CS[layer],1],
             padding = 'SAME',
             name = 'cov2d')

          # activation after either batch normalization or bias add
          if use_batch_normalization:
             L = tf.nn.relu(batch_normalization(inputs=L))
          else:
             b = tf.Variable(tf.random_normal([C[layer+1]]))
             L = tf.nn.relu(L + b)

          # max pooling
          L = tf.nn.max_pool(L, 
             ksize=[1, PK[layer], PK[layer], 1], 
             strides=[1, PS[layer], PS[layer], 1], 
             padding='SAME')

          # drop-out
          if use_dropout:
             L = tf.nn.dropout(L, keep_prob=keep_prob)
      
          if enable_summary:
             if not use_batch_normalization:
                tf.summary.histogram("bias", b)
             tf.summary.histogram("weights", W)
             tf.summary.histogram("layer", L)

   L = tf.reshape(L, [-1, 128 * 4 * 4])
   
   with tf.variable_scope('fc0'):
       W = tf.get_variable("W", shape=[128 * 4 * 4, 625],
                               regularizer=w_regularizer,
                               initializer=w_initializer)

       if use_batch_normalization:
          L = tf.matmul(L, W)
          L = tf.nn.relu(batch_normalization(inputs=L))
       else:
          b = tf.Variable(tf.random_normal([625]))
          L = tf.nn.relu(tf.matmul(L, W) + b)
       if use_dropout:
          L = tf.nn.dropout(L, keep_prob=keep_prob)
      
       if enable_summary:
          if not use_batch_normalization:
             tf.summary.histogram("bias", b)
          tf.summary.histogram("weights", W)
          tf.summary.histogram("layer", L)

   with tf.variable_scope('fc1'):
       W = tf.get_variable("W", shape=[625,n_classes],
                               regularizer=w_regularizer,
                               initializer=w_initializer)

       b = tf.Variable(tf.random_normal([n_classes]))
       L = tf.matmul(L, W) + b
      
       if enable_summary:
          tf.summary.histogram("bias", b)
          tf.summary.histogram("weights", W)
          tf.summary.histogram("layer", L)

hypothesis = tf.identity(L,name='hypothesis')

if is_training:
   # define cost/loss & optimizer
   cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
       logits=hypothesis, labels=Y))
   
   # for Batch Normalization
   if use_batch_normalization:
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
         optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
   else:
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
   
   if enable_summary:
      tf.summary.scalar("loss", cost)
   
   last_epoch = tf.Variable(0, name='last_epoch')
   
   # Summary
   if enable_summary:
      summary = tf.summary.merge_all()
   
   # initialize
   sess = tf.Session()
   sess.run(tf.global_variables_initializer())
   
   # Create summary writer
   if enable_summary:
      writer = tf.summary.FileWriter(TB_SUMMARY_DIR)
      writer.add_graph(sess.graph)
   
   global_step = 0
else:
   sess = tf.Session()
   sess.run(tf.global_variables_initializer())

# Saver and Restore
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECK_POINT_DIR)

if checkpoint and checkpoint.model_checkpoint_path:
    try:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
        if is_training:
           tf.train.export_meta_graph(filename='mnist_cnn.metatext',as_text=True)
    except:
        print("Error on loading old network weights")
else:
    print("Could not find old network weights")

if not is_training:
   saver.save(sess, CHECK_POINT_DIR + "/converted")
   exit()


start_from = sess.run(last_epoch)

# train my model
print('Start learning from:', start_from)

for epoch in range(start_from, start_from+training_epochs):
    print('Start Epoch:', epoch)

    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_ximage = np.reshape(batch_xs, [-1, 28, 28, 1])
        #feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
        feed_dict = {X: batch_ximage, Y: batch_ys, keep_prob: 0.7} if use_dropout else {X: batch_ximage, Y: batch_ys}
        l, __ = sess.run([cost,optimizer], feed_dict=feed_dict)
        global_step += 1
        avg_cost += l / total_batch

        if enable_summary==True and i%1000==0:
            s, _ = sess.run([summary, optimizer], feed_dict=feed_dict)
            writer.add_summary(s, global_step=global_step)

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print("Saving network...")
    sess.run(last_epoch.assign(epoch + 1))
    if not os.path.exists(CHECK_POINT_DIR):
        os.makedirs(CHECK_POINT_DIR)
    saver.save(sess, CHECK_POINT_DIR + "/model", global_step=i)

    tf.train.export_meta_graph(filename='mnist_cnn.metatext',as_text=True)

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, 
      feed_dict={X: np.reshape(mnist.test.images,[-1,28,28,1]), Y: mnist.test.labels, keep_prob: 1} if use_dropout else {X: np.reshape(mnist.test.images,[-1,28,28,1]), Y: mnist.test.labels} 
      ))

# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1), 
    feed_dict={X: np.reshape(mnist.test.images[r:r + 1],[-1,28,28,1]), keep_prob: 1} if use_dropout else {X: np.reshape(mnist.test.images[r:r + 1],[-1,28,28,1])}
    ))

# plt.imshow(mnist.test.images[r:r + 1].
#           reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()

'''

...

Successfully loaded: ./tb/mnist1/model-549
Start learning from: 2
Epoch: 2

...
tensorboard --logdir tb/
Starting TensorBoard b'41' on port 6006
(You can navigate to http://10.0.1.4:6006)

'''
