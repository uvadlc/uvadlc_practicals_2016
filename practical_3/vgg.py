import tensorflow as tf
import numpy as np

VGG_FILE = './pretrained_params/vgg16_weights.npz'

def load_pretrained_VGG16_pool5(input, scope_name='vgg'):
    """
    Load an existing pretrained VGG-16 model.
    See https://www.cs.toronto.edu/~frossard/post/vgg16/

    Args:
        input:         4D Tensor, Input data
        scope_name:    Variable scope name

    Returns:
        pool5: 4D Tensor, last pooling layer
        assign_ops: List of TF operations, these operations assign pre-trained values
                    to all parameters.
    """

    with tf.variable_scope(scope_name):

        vgg_weights, vgg_keys = load_weights(VGG_FILE)

        assign_ops = []
        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            vgg_W = vgg_weights['conv1_1_W']
            vgg_B = vgg_weights['conv1_1_b']
            kernel = tf.get_variable('conv1_1/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv1_1/' + "biases", vgg_B.shape,
                initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv1_1 = tf.nn.relu(out, name=scope)


        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            vgg_W = vgg_weights['conv1_2_W']
            vgg_B = vgg_weights['conv1_2_b']
            kernel = tf.get_variable('conv1_2/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))

            conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv1_2/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv1_2 = tf.nn.relu(out, name=scope)

        # pool1
        pool1 = tf.nn.max_pool(conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            vgg_W = vgg_weights['conv2_1_W']
            vgg_B = vgg_weights['conv2_1_b']
            kernel = tf.get_variable('conv2_1/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv2_1/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv2_1 = tf.nn.relu(out, name=scope)

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            vgg_W = vgg_weights['conv2_2_W']
            vgg_B = vgg_weights['conv2_2_b']
            kernel = tf.get_variable('conv2_2/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv2_2/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv2_2 = tf.nn.relu(out, name=scope)

        # pool2
        pool2 = tf.nn.max_pool(conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            vgg_W = vgg_weights['conv3_1_W']
            vgg_B = vgg_weights['conv3_1_b']
            kernel = tf.get_variable('conv3_1/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv3_1/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv3_1 = tf.nn.relu(out, name=scope)

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            vgg_W = vgg_weights['conv3_2_W']
            vgg_B = vgg_weights['conv3_2_b']
            kernel = tf.get_variable('conv3_2/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer()
                                     )

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv3_2/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv3_2 = tf.nn.relu(out, name=scope)

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            vgg_W = vgg_weights['conv3_3_W']
            vgg_B = vgg_weights['conv3_3_b']
            kernel = tf.get_variable('conv3_3/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv3_3/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv3_3 = tf.nn.relu(out, name=scope)

        # pool3
        pool3 = tf.nn.max_pool(conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            vgg_W = vgg_weights['conv4_1_W']
            vgg_B = vgg_weights['conv4_1_b']
            kernel = tf.get_variable('conv4_1/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv4_1/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv4_1 = tf.nn.relu(out, name=scope)

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            vgg_W = vgg_weights['conv4_2_W']
            vgg_B = vgg_weights['conv4_2_b']
            kernel = tf.get_variable('conv4_2/'  + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv4_2/'  + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv4_2 = tf.nn.relu(out, name=scope)

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            vgg_W = vgg_weights['conv4_3_W']
            vgg_B = vgg_weights['conv4_3_b']
            kernel = tf.get_variable('conv4_3/'  + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv4_3/'  + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv4_3 = tf.nn.relu(out, name=scope)

        # pool4
        pool4 = tf.nn.max_pool(conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            vgg_W = vgg_weights['conv5_1_W']
            vgg_B = vgg_weights['conv5_1_b']
            kernel = tf.get_variable('conv5_1/'  + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv5_1/'  + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv5_1 = tf.nn.relu(out, name=scope)


        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            vgg_W = vgg_weights['conv5_2_W']
            vgg_B = vgg_weights['conv5_2_b']
            kernel = tf.get_variable('conv5_2/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv5_2/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv5_2 = tf.nn.relu(out, name=scope)


        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            vgg_W = vgg_weights['conv5_3_W']
            vgg_B = vgg_weights['conv5_3_b']
            kernel = tf.get_variable('conv5_3/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv5_3/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv5_3 = tf.nn.relu(out, name=scope)


        # pool5
        pool5 = tf.nn.max_pool(conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool5')
        print("pool5.shape: %s" % pool5.get_shape())

    return pool5, assign_ops

def load_weights(weight_file):
  weights = np.load(weight_file)
  keys = sorted(weights.keys())
  return weights, keys
