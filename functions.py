import tensorflow as tf
import numpy as np
# preprocessing functions

# for global contrast normalization
def gcn(x, s=1, l=0, e=10**(-8)):
#     transpose x(NHWC)->x(CHWN)
    [N, H, W, C] = x.shape
    x = x.transpose([3,1,2,0])
    mean = (np.ones([H, W, N]) * (np.ones([W, N]) * np.mean(a=x, axis=(0,1,2))))
    div = np.sqrt(l + np.sum(a=np.square(x - mean), axis=(0,1,2))/(C * W * H))
#     implement max(e, xi) elementwise in tensor
    div[div < e] = e
    ret = (x - mean) / (np.ones([H, W, N]) * (np.ones([W, N]) * div))
#     transpose back to (NHWC)
    return ret.transpose([3,1,2,0])

# zca whitening
def zca(x, e=0.1):
    x_white = np.reshape(x, (-1, x.shape[0]), 'C')
    [U, S, V] = np.linalg.svd(np.dot(x_white, x_white.transpose()) / x_white.shape[0])
    x_white =np.dot(U, np.dot(np.diag(1 / np.sqrt(S + e)), np.dot(U.transpose(), x_white)))
    return np.reshape(x_white, x.shape, 'C')



# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding = 'SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# extract patches from feature maps
# input shape N, H, W, C
# output shape N, H, W, K, C
def extract_patches(x, padding, ksize=2, stride=2):
    temp = tf.extract_image_patches(images=x, ksizes=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], rates=[1,1,1,1], padding=padding)
    [N, H, W, C] = temp.get_shape().as_list()
    C = x.get_shape().as_list()[-1]
#     reshape to N,H,W,K,C
    temp = tf.reshape(temp, [-1, H, W, ksize*ksize, C])
    return temp

# cast function use to compute the frequency
def cast(p):
    return tf.to_int32(tf.round(p))

# compute the frequency of element in each patch
# input extracted patches tensor in shape N, H, W, K, C
# output frequency tensor in shape N, H, W, K, C
def majority_frequency(temp):
    [N, H, W, K, C] = temp.get_shape().as_list()

    temp = cast(temp)
#     build one hot vector
    temp = tf.transpose(temp, [0,1,2,4,3])
    one_hot = tf.one_hot(indices=temp, depth=tf.reduce_max(temp) + 1, dtype=tf.float32)
#     the dimension is bathch, row, col, lay, one hot
#     the order tensorflow takes, when doiong transpose, it will from the most right to most left
    one_hot = tf.reduce_sum(one_hot, axis=4)
    temp = tf.transpose(temp, [0, 3, 1, 2, 4])
    temp = tf.reshape(temp, [N*H*W*C*K,1])
    one_hot = tf.transpose(one_hot, [0,3,1,2,4])
    one_hot = tf.reshape(one_hot, [N*H*W*C, -1])
    
    index = tf.constant(np.array([range(temp.get_shape().as_list()[0])])/ K, dtype=tf.int32)
    temp = tf.concat((tf.transpose(index), temp), axis=1)
    
#     to get the percentage
    temp = tf.gather_nd(one_hot, temp)
    temp = tf.reshape(temp, [N, C, H, W, K])
#     finally we change it back to N,H,W,K,C
    temp = tf.transpose(temp, [0, 2, 3, 4, 1])
    return temp

# compute weight based on frequency tensor
# fun could be tf.reduce_max, tf.reduce_sum, reduce_size(in str)
# output in shape N, H, W, K, C
def compute_weight(w, fun):
    if isinstance(fun, str): deno = w.get_shape().as_list()[3]
    else: deno = fun(w, axis=3, keep_dims=True)
    temp = tf.divide(w, deno)
    return temp


# ---------------------------------- pooling function ---------------------------------
# the mask have shape of N, H, W, K C
def max_pool(p):
    return tf.reduce_max(p, axis=3)

def max_pool_with_mask(p):
    pool = max_pool(p)
    [N, H, W, K, C] = p.get_shape().as_list()
    mask = tf.reshape(pool, [N, H, W, 1, C])
    mask = tf.cast(tf.equal(p, mask), dtype = tf.float32)
    mask = tf.div(mask, tf.reduce_sum(mask, axis=3, keep_dims=True))
    return pool, mask

# for majority pooling if the maximum frequency of one window is 1, then we pool the max from this window
def majority_pool(p, f):
    pool, mask = majority_pool_with_mask(p, f)
    return pool

def majority_pool_with_mask(p,f):
    btemp = tf.reduce_max(f , axis=[3], keep_dims=True)
#     get the index of the majority element
    pool = tf.cast(tf.equal(f, btemp), dtype=tf.float32)
#     use the largest frequency to represent each window
    btemp = tf.squeeze(btemp, squeeze_dims=3)
#     compute mean of the elements that have same round value in each window
    pool = tf.div(tf.reduce_sum(tf.multiply(p, pool), axis=[3]), btemp)
#     when the largest frequency is 1, then we just the max value in p as the result, else use the mean of the of elements
#     having the same round value, as the result.
    pool = tf.where(tf.equal(btemp, 1), tf.reduce_max(p, axis=[3]), pool)

    [N, H, W, K, C] = p.get_shape().as_list()
    mask = tf.reshape(pool, [N, H, W, 1, C])
    mask = tf.cast(tf.equal(cast(p), cast(mask)), dtype = tf.float32)
    mask = tf.div(mask, tf.reduce_sum(mask, axis=3, keep_dims=True))
    return pool, mask

# pcaPool
# if m == 1, then consider each window as an unique instances, and each window have their own pca encoder
# if m != 1, then all windows fetch from the same feature map share one pca encoder
def pca_pool(p, m = 1):
    pool, mask = pca_pool_with_mask(p, m)
    return pool

def pca_pool_with_mask(temp, m = 1):
    [N, H, W, K, C] = temp.get_shape().as_list()
    if m == 1:
        temp = tf.transpose(temp, [0,1,2,4,3])
        temp = tf.reshape(temp, [-1, K, 1])
    else:
        temp = tf.transpose(temp, [0,4,3,1,2])
        temp = tf.reshape(temp, [-1, K, H*W])
#     compute for svd
    [s, u, v] = tf.svd(tf.matmul(temp, tf.transpose(temp, [0,2,1])), compute_uv=True)
#     use mark to remove Eigenvector except for the first one, which is the main component
    temp_mark = np.zeros([K,K])
    temp_mark[:,0] = 1
    mark = tf.constant(temp_mark, dtype=tf.float32)
    
#     after reduce_sum actually it has been transposed automatically
    u = tf.reduce_sum(tf.multiply(u, mark), axis=2)
    u = tf.reshape(u, [-1, 1, K])
    u = u / np.sqrt(K)
    # divide sqrt(k) to remove the effect of size of window
    temp = tf.matmul(u, temp)/np.sqrt(K)
    if m == 1: 
        temp = tf.reshape(temp, [-1, H, W, C])
        u = tf.transpose(tf.reshape(u, [N, H, W, C, K]), [0, 1, 2, 4, 3])
    else: 
        temp = tf.reshape(temp, [-1, C, H, W])
        temp = tf.transpose(temp, [0, 2, 3, 1])
        u = tf.transpose(tf.reshape(u, [N, C, K, 1, 1]), [0, 3, 4, 2, 1])
        u = tf.multiply(u, tf.ones_like(y))
    return temp, u

# weithed pooling functions
# weight before maxpool p:= patches, w:= weights
def weight_pool(p, f, reduce_fun, pool_fun):
    temp, u = weight_pool_with_mask(p, f, reduce_fun, pool_fun)


def weight_pool_with_mask(p, f, reduce_fun, pool_fun):
    w = compute_weight(f, reduce_fun)
    temp = tf.multiply(p, w)
    if pool_fun is majority_pool_with_mask:
        temp, u = pool_fun(temp, majority_frequency(temp))
    else: temp, u = pool_fun(temp)
    u = tf.multiply(u, w)
    return temp, u

# weight is used to help to decide, but pool the original number
def weight_pool_original_with_mask(p, f, reduce_fun, pool_fun):
    pool, mask = weight_pool_with_mask(p, f, reduce_fun, pool_fun)
    mask = tf.cast(tf.greater(mask, 0), dtype=tf.float32)
    mask = tf.div(mask, tf.reduce_sum(mask, axis=3, keep_dims=True))
    return tf.multiply(p, mask)

def weight_pool_original(p, f, reduce_fun, pool_fun):
    pool, mask = weight_pool_original(p, f, reduce_fun, pool_fun)
    return pool

# maxpool before weight
def pool_weight(p, f, reduce_fun, pool_fun):
    pool, mask = pool_weight_with_mask(p, f, reduce_fun, pool_fun)
    return pool

def pool_weight_with_mask(p, f, reduce_fun, pool_fun):
    #     for now both p and w are in the shape of N,H,W,K,C
    [N, H, W, K, C] = p.get_shape().as_list()
    w = compute_weight(f, reduce_fun)
    if pool_fun is majority_pool_with_mask:
        p, u = pool_fun(p, f)
        w = tf.reduce_max(w, axis=3)
    else:
#     argmax in the shape of N, H, W, C
        argmax = tf.argmax(p, axis=3)
        p, u = pool_fun(p)
#     move C before H
        argmax = tf.transpose(argmax, [0, 3, 1, 2])
        w = tf.transpose(w, [0, 4, 1, 2, 3])
#     flatten argmax and w
        argmax = tf.reshape(argmax, [N*H*W*C, 1])
        w = tf.reshape(w, [N*H*W*C, K])
#     create index helper
        index = tf.constant(np.array([range(argmax.get_shape().as_list()[0])]), dtype=tf.int64)
        argmax = tf.concat((tf.transpose(index), argmax), axis=1)
#     get the corresponding weight of the max
        w = tf.gather_nd(w, argmax)
        w = tf.reshape(w, [N, C, H, W])
        w = tf.transpose(w, [0, 2, 3, 1])
    p = tf.multiply(p, w)
    u = tf.multiply(u, w)
    return p, u



# compute filter weight gradient
# for this exp we use 5*5 as the kernel
def filter_gradient(e, pre, cur):
    [N, H, W, C] = cur.get_shape().as_list()
    pre = tf.pad(pre, tf.constant([[0,0],[2,2],[2,2],[0,0]]))
    pre = extract_patches(pre, 'VALID', H, 1)
    pre = tf.reshape(pre, [N, 5, 5, H * H, -1, 1])
    grad_w = tf.reduce_sum(tf.multiply(pre, tf.reshape(e, [N, 1, 1, H * H, 1, C])), axis=[3, 0]) / N
    grad_b = tf.reduce_sum(e, axis=[0,1,2]) / N
    return grad_w, grad_b

# transmit error from conv layer to pooling layer
def error_conv2pooling(e, w):
    return tf.nn.conv2d(e, tf.reverse(tf.transpose(w, [0, 1, 3, 2]), axis=[0, 1]), 
        strides=[1, 1, 1, 1], padding = 'SAME')


# global variable
# unpooling_method = {'max': max_unpooling_mark}

# compute error from pooling to conv layer
def error_pooling2conv(e, mask):
    [N, H, W, K, C] = mask.get_shape().as_list()
    e = tf.multiply(mask, tf.reshape(e, [N, H, W, 1, C]))
    e = tf.reshape(e, [N, -1, K, C])
    e = tf.extract_image_patches(images=e, ksizes=[1, H, int(np.sqrt(K)), 1], 
        strides=[1, H, int(np.sqrt(K)), 1], padding="VALID", rates=[1, 1, 1, 1])
    e = tf.reshape(e, [N, H * int(np.sqrt(K)), W * int(np.sqrt(K)), C])
    return e
