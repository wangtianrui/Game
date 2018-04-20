import tensorflow as tf

size = 37
def getWeight(shape):
    w = tf.truncated_normal(shape, 0.1)
    return tf.Variable(w)


def getBias(shape):
    b = tf.constant(0.1, shape=shape)
    return tf.Variable(b)


def fc(input, keep_prob):
    fc1_w = getWeight([size, 128])
    fc1_b = getBias([128])
    fc1 = tf.nn.relu(tf.matmul(input, fc1_w) + fc1_b)

    fc2_w = getWeight([128, 512])
    fc2_b = getBias([512])
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_w) + fc2_b)

    h_fc1_drop = tf.nn.dropout(fc2, keep_prob)
    fc3_w = getWeight([512, 2])
    fc3_b = getBias([2])
    fc3 = tf.matmul(h_fc1_drop, fc3_w) + fc3_b

    return fc3, fc1_w, fc2_w



def fc2(input, keep_prob):
    fc1_w = getWeight([size, 128])
    fc1_b = getBias([128])
    fc1 = tf.nn.relu(tf.matmul(input, fc1_w) + fc1_b)

    h_fc1_drop = tf.nn.dropout(fc1, keep_prob)
    fc3_w = getWeight([128, 2])
    fc3_b = getBias([2])
    fc3 = tf.matmul(h_fc1_drop, fc3_w) + fc3_b

    return fc3
