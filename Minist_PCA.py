import tensorflow as tf
import numpy as np
import functions as func
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 300000
batch_size = 128
mini_batch = 16
train_iter = batch_size//mini_batch
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units


# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
para = [weights['wc1'], weights['wc2'], weights['wd1'], weights['out'],
        biases['bc1'], biases['bc2'], biases['bd1'], biases['out']]


# body of network
def body(i, x, y, grads):
    # Convolution Layer
    inputx = x.read(index=i)
    conv1 = func.conv2d(inputx, weights['wc1'], biases['bc1'])
    # Pooling (down-sampling)
    p1 = func.extract_patches(conv1, 'SAME', 2, 2)
    f1 = func.majority_frequency(p1)
    #     maxpooling
    pool1, mask1 = func.pca_pool_with_mask(temp=p1)

    # Convolution Layer
    conv2 = func.conv2d(pool1, weights['wc2'], biases['bc2'])
    #     Pooling (down-sampling)
    p2 = func.extract_patches(conv2, 'SAME', 2, 2)
    f2 = func.majority_frequency(p2)
    #     maxpooling
    pool2, mask2 = func.pca_pool_with_mask(temp=p2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc = tf.reshape(pool2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    yi = y.read(index=i)
    temp_pred = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    grads[8] = pred.write(index=i, value=temp_pred)

    # ------------------------------end of define graph------------------------------------

    # ------------------------------define gradient descent-------------------------

    # the last fc
    e = tf.nn.softmax(temp_pred) - yi
    grads[3] = grads[3].write(index=i, value=tf.transpose(fc1) @ e)
    grads[7] = grads[7].write(index=i, value=tf.reduce_sum(e, axis=0))

    # the second last fc
    # we use droupout at the last second layer, then we should just update the nodes that are active
    e = tf.multiply(e @ tf.transpose(weights['out']), tf.cast(tf.greater(fc1, 0), dtype=tf.float32)) / dropout
    grads[2] = grads[2].write(index=i, value=tf.transpose(fc) @ e)
    grads[6] = grads[6].write(index=i, value=tf.reduce_sum(e, axis=0))

    # the last pooling layer
    e = e @ tf.transpose(weights['wd1'])
    e = tf.reshape(e, pool2.get_shape().as_list())

    # the last conv layer
    # unpooling get error from pooling layer
    e = func.error_pooling2conv(e, mask2)

    # multiply with the derivative of the active function on the conv layer
    #     this one is also important this is a part from the upsampling, but
    e = tf.multiply(e, tf.cast(tf.greater(conv2, 0), dtype=tf.float32))
    temp1, temp2 = func.filter_gradient(e, pool1, conv2)
    grads[1] = grads[1].write(index=i, value=temp1)
    grads[5] = grads[5].write(index=i, value=temp2)

    # conv to pool
    e = func.error_conv2pooling(e, weights['wc2'])

    # pool to the first conv
    e = func.error_pooling2conv(e, mask1)
    e = tf.multiply(e, tf.cast(tf.greater(conv1, 0), dtype=tf.float32))
    temp1, temp2 = func.filter_gradient(e, inputx, conv1)
    grads[0] = grads[0].write(index=i, value=temp1)
    grads[4] = grads[4].write(index=i, value=temp2)
    i += 1
    return i, x, y, grads



# compute gradient and update the weights
xs = tf.reshape(x, shape=[batch_size, 28, 28, 1])
inputxs = tf.TensorArray(dtype=tf.float32, size=train_iter, clear_after_read=True).split(xs, [mini_batch] * train_iter)
ys = tf.TensorArray(dtype=tf.float32, size=train_iter, clear_after_read=True).split(y, [mini_batch] * train_iter)

grad_k_1 = tf.TensorArray(dtype=tf.float32, size=train_iter)
grad_k_2 = tf.TensorArray(dtype=tf.float32, size=train_iter)
grad_w_3 = tf.TensorArray(dtype=tf.float32, size=train_iter)
grad_w_out = tf.TensorArray(dtype=tf.float32, size=train_iter)
grad_b_1 = tf.TensorArray(dtype=tf.float32, size=train_iter)
grad_b_2 = tf.TensorArray(dtype=tf.float32, size=train_iter)
grad_b_3 = tf.TensorArray(dtype=tf.float32, size=train_iter)
grad_b_out = tf.TensorArray(dtype=tf.float32, size=train_iter)
pred = tf.TensorArray(dtype=tf.float32, size=train_iter)
grads = [grad_k_1, grad_k_2, grad_w_3, grad_w_out, grad_b_1, grad_b_2, grad_b_3, grad_b_out, pred]

# gradient
i0 = tf.constant(0)
con = lambda i, x, y, g: i < train_iter

i0, inputx, ys, grads = tf.while_loop(cond=con, body=body, loop_vars=[i0, inputxs, ys, grads], back_prop=False, parallel_iterations=1)
pred = grads[-1].stack()
del grads[-1]
for i in range(len(grads)):
    grads[i] = (tf.reduce_mean(grads[i].stack(), axis=0), para[i])

pred = tf.reshape(pred, [-1, 10])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).apply_gradients(grads)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1), tf.argmax(y,1)), tf.float32))




config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
with tf.Session(config=config) as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
#     while step < 2:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(opt, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            cos, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                y: batch_y, keep_prob: 1.})

            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(cos) +
                  "\nTraining Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

#     Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:batch_size], y: mnist.test.labels[:batch_size],
keep_prob: 1.}))
