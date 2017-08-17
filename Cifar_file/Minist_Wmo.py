import tensorflow as tf
import numpy as np
import functions as func

# Parameters
learning_rate = 0.001
training_iters = 300000
batch_size = 128
mini_batch = 8
train_iter = batch_size//mini_batch
display_step = 10

# Network Parameters
H = 32
W = 32
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# load data
data_x = np.load('../cifar_x.npy')
data_y = np.load('../cifar_y.npy')
test_x = np.load('../cifar_test_x.npy')
test_y = np.load('../cifar_test_y.npy')
index = 0

# tf Graph input
x = tf.placeholder(tf.float32, [None, H, W, 3])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([8*8*64, 1024])),
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
def body2(i, x, out):
    # Convolution Layer
    inputx = x.read(index=i)
    conv1 = func.conv2d(inputx, weights['wc1'], biases['bc1'])
    # Pooling (down-sampling)
    p1 = func.extract_patches(conv1, 'SAME', 2, 2)
    f1 = func.majority_frequency(p1)
    #     maxpooling
    pool1 = func.weight_pool_original(p=p1, f=f1, reduce_fun=tf.reduce_max, pool_fun=func.max_pool_with_mask)

    # Convolution Layer
    conv2 = func.conv2d(pool1, weights['wc2'], biases['bc2'])
    #     Pooling (down-sampling)
    p2 = func.extract_patches(conv2, 'SAME', 2, 2)
    f2 = func.majority_frequency(p2)
    #     maxpooling
    pool2 = func.weight_pool(p=p2, f=f2, reduce_fun=tf.reduce_max, pool_fun=func.max_pool_with_mask)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc = tf.reshape(pool2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    # Output, class prediction
    out = out.write(index=i, value=tf.add(tf.matmul(fc1, weights['out']), biases['out']))
    i += 1
    return i, x, out


inputxs = tf.TensorArray(dtype=tf.float32, size=train_iter, clear_after_read=True).split(x, np.array([mini_batch] * train_iter))
out = tf.TensorArray(dtype=tf.float32, size=train_iter)


# compute gradient and update the weights
i0 = tf.constant(0)
con = lambda i, x, g: i < train_iter

i0, inputx, out = tf.while_loop(cond=con, body=body2, loop_vars=[i0, inputxs, out], parallel_iterations=1)

pred = tf.reshape(out.stack(), [-1, 10])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1), tf.argmax(y,1)), tf.float32))



cost_save=[]
accuracy_save=[]
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
        batch_x, batch_y, index = func.get_batch(batch_size, data_x, data_y, index)
        # Run optimization op (backprop)
        sess.run(opt, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            cos, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                y: batch_y, keep_prob: 1.})
            cost_save.append(cos)
            accuracy_save.append(acc)
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(cos) +
                  "\nTraining Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

#     Calculate accuracy for 256 mnist test images
    index = 0
    batch_x, batch_y, index = func.get_batch(batch_size, test_x, test_y, index)
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.}))
np.save('../output/cwmo_cost',np.array(cost_save))
np.save('../output/cwmo_acc', np.array(accuracy_save))
