import tensorflow as tf
import pandas as pd
import numpy as np
import random

num_class = 2
batch_size = 33
train_path = r"./data/lihao_data.csv"
test_path = r"./data/lihaotest.csv"
learning_rate = 0.0001
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
size = 63


def readData(path):
    whole_data = pd.read_csv(path, encoding='utf-8')
    whole_data = whole_data.fillna(9999)
    whole_list = np.array(whole_data)
    print(whole_list)
    return whole_list


def makeData(data):
    x = []
    y = []
    ran = []
    # print(len(data))
    # print(len(data[1480]))
    # print(data[1480])
    for i in range(33):
        index = random.randint(0, 6585)
        # print("test data len", len(data))
        row = data[index]
        label = [int(row[46])]
        y.append(label)
        row = np.delete(row, 46)
        x.append(row)
        ran.append(index)
    # print(label)
    # print("lenx:",len(x))
    # print("len x1:",len(x[1]))
    # x = tf.reshape(x, shape=[batch_size, size])
    # label = tf.reshape(label, shape=[batch_size, 2])
    return x, y, ran


def getWeight(shape):
    w = tf.truncated_normal(shape, 0.1)
    return tf.Variable(w)


def getBias(shape):
    b = tf.constant(0.1, shape=shape)
    return tf.Variable(b)


def fc(input, keep_prob):
    fc1_w = getWeight([size, 64])
    fc1_b = getBias([64])
    fc1 = tf.nn.relu(tf.matmul(input, fc1_w) + fc1_b)
    h_fc1_drop = tf.nn.dropout(fc1, keep_prob)
    fc2_w = getWeight([64, 2])
    fc2_b = getBias([2])
    fc2 = tf.matmul(h_fc1_drop, fc2_w) + fc2_b

    return fc2, fc1_w, fc2_w


def cross_entropy(fc_result, label):
    label = tf.one_hot(label, depth=num_class)
    return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=fc_result, labels=label))


def train():
    x_ = tf.placeholder(shape=[batch_size, size], dtype="float")
    y_ = tf.placeholder(shape=[batch_size, 1], dtype="int32")
    keep_prob = tf.placeholder("float")

    global_step = tf.Variable(0, name="global_step", trainable=False)
    logits, fc1, fc2 = fc(x_, keep_prob)
    loss = cross_entropy(logits, y_)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    init = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    sess.run(init)
    saver = tf.train.Saver(tf.global_variables())
    whole_data = readData(train_path)
    # whole_data = sess.run(read_data)
    for i in range(30000):
        if i % 100 == 0:
            x, y, indexs = makeData(whole_data)
            train_accuracy = sess.run(accuracy, feed_dict={x_: x, y_: y, keep_prob: 1})
            lo = sess.run(loss, feed_dict={x_: x, y_: y, keep_prob: 1})
            f1w = sess.run(fc1, feed_dict={x_: x, y_: y, keep_prob: 1})
            f2w = sess.run(fc2, feed_dict={x_: x, y_: y, keep_prob: 1})
            #print("f1:",f1w,"\n","f2:",f2w)
            print("step %d, train accuracy %g , loss %g" % (i, train_accuracy, lo))
            #print(indexs)
        elif i == 2999:
            path = saver.save(sess, "E:/python_programes/MyMNIST/result/model.ckpt", global_step=i)
            if (path == None):
                print("path is none")
            else:
                print(path)
        sess.run(train_op, feed_dict={x_: x, y_: y, keep_prob: 0.5})


if __name__ == "__main__":
    train()
