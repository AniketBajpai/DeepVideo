import tensorflow as tf
from tensorflow.python.framework import ops

# ==================================
# ---------- ACTIVATIONS --------- #
# ==================================


def lrelu(x, leak=0.2, name="lrelu"):
    """ leaky relu: maximum(x, leak*x) """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

# ==================================
# ---------- LOSS FUNCTIONS --------- #
# ==================================


def l1Penalty(x, scale=0.1, name="l1_penalty"):
    l1P = tf.contrib.layers.l1_regularizer(scale)
    return l1P(x)


def huber_loss(labels, predictions, delta=1.0):
    """ Huber loss: L2 befor delta, L1 after delta """
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.
    Let `x = `, `z = targets`.  The logistic loss is
        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) + (1. - targets) * tf.log(1. - preds + eps)))

# ==================================
# ---------- LAYER MAPS --------- #
# ==================================


def linear(input_, output_size, name='linear', stddev=0.01, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])


def conv2d(input, output_shape, is_train,
           k=(5, 5), s=(2, 2), stddev=0.01,
           name="conv2d"):
    k_h, k_w = k
    s_h, s_w = s
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input.get_shape()[-1], output_shape[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input, w, strides=[1, s_h, s_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        bn = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9,
                                          is_training=is_train, updates_collections=None)
        out = lrelu(bn)
        return out


def deconv2d(input, output_shape, is_train,
             k=(5, 5), s=(2, 2), stddev=0.01, activation_fn='relu',
             name="deconv2d", with_w=False):
    k_h, k_w = k
    s_h, s_w = s
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', [k_h, k_w, output_shape[-1], input.get_shape()[-1]],
                                  initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input, weights, output_shape=output_shape,
                                        strides=[1, s_h, s_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if activation_fn == 'relu':
            deconv = tf.contrib.layers.batch_norm(deconv, center=True, scale=True,
                                                  decay=0.9, is_training=is_train, updates_collections=None)
            deconv = tf.nn.relu(deconv)
        elif activation_fn == 'tanh':
            deconv = tf.nn.tanh(deconv)
        else:
            raise ValueError('Invalid activation function.')

        if with_w:
            return deconv, weights, biases
        else:
            return deconv


def conv3d(input_, output_shape, is_train,
           k=(4, 4, 4), d=(2, 2, 2), pad=(1, 1, 1), stddev=0.01,
           name="conv3d"):
    k_t, k_h, k_w = k
    d_t, d_h, d_w = d
    pad_t, pad_h, pad_w = pad
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_t, k_h, k_w, input_.get_shape()[-1], output_shape[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv3d(input_, w, strides=[1, d_t, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        bn = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9,
                                          is_training=is_train, updates_collections=None)
        out = lrelu(bn)
        return out


def deconv3d(input_, output_shape, is_train,
             k=(4, 4, 4), d=(2, 2, 2), pad=(1, 1, 1), stddev=0.01, activation_fn='relu',
             name="deconv3d", with_w=False):
    k_t, k_h, k_w = k
    d_t, d_h, d_w = d
    pad_t, pad_h, pad_w = pad
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        weights = tf.get_variable('weights', [k_t, k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                  initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv3d_transpose(input_, weights, output_shape=output_shape,
                                        strides=[1, d_t, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), output_shape)

        if activation_fn == 'relu':
            deconv = tf.contrib.layers.batch_norm(deconv, center=True, scale=True,
                                                  decay=0.9, is_training=is_train, updates_collections=None)
            deconv = tf.nn.relu(deconv)
        elif activation_fn == 'tanh':
            deconv = tf.nn.tanh(deconv)
        else:
            raise ValueError('Invalid activation function.')

        if with_w:
            return deconv, weights, biases
        else:
            return deconv
