import tensorflow as tf
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import os
from sklearn import preprocessing

num_class = 2
batch_size = 20
train_path = r"./data/wang_data.csv"
test_path = r"./data/wang_data_test.csv"
learning_rate = 0.0000001
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
size = 37
log_path = r"./log/"


def readData(path):
    whole_data = pd.read_csv(path, encoding='utf-8')
    whole_data = whole_data.fillna(0)
    whole_list = np.array(whole_data)
    for i in range(len(whole_list)):
        for temp in range(len(whole_list[i])):
            if whole_list[i][temp] == " ":
                whole_list[i][temp] = 0
    # print("转换前：", whole_list)
    whole_list = np.array(whole_list, dtype="float")
    # print(whole_list)
    return whole_list


def makeData(data, flag):
    x = []
    y = []
    ran = []
    # print(len(data))
    # print(len(data[1480]))
    # print(data)
    if flag == "train":
        for i in range(batch_size):
            index = random.randint(0, 6384)
            # print("test data len", len(data))
            row = data[index]
            label = [int(row[37])]
            y.append(label)
            row = np.delete(row, 37)
            x.append(row)
            ran.append(index)
    else:
        for i in range(batch_size):
            index = random.randint(0, 66)
            # print("test data len", len(data))
            row = data[index]
            label = [int(row[37])]
            y.append(label)
            row = np.delete(row, 37)
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
    fc1_w = getWeight([size, 128])
    fc1_b = getBias([128])
    fc1 = tf.nn.relu(tf.matmul(input, fc1_w) + fc1_b)
    h_fc1_drop = tf.nn.dropout(fc1, keep_prob)
    fc2_w = getWeight([128, 2])
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
    test_data = readData(test_path)
    # whole_data = sess.run(read_data)
    for i in range(3000000):
        if i % 100 == 0:
            x, y, indexs = makeData(whole_data, "train")
            test_x, test_y, test_index = makeData(test_data, "test")

            #归一化
            min_max_scaler = preprocessing.MinMaxScaler()
            x = min_max_scaler.fit_transform(x)

            # 标准化
            ss_x = StandardScaler()
            ss_y = StandardScaler()
            x = ss_x.fit_transform(x)
            y = ss_y.fit_transform(y)
            print("x:", x)
            test_x = ss_x.transform(test_x)
            #print("test x:", x)
            test_y = ss_y.transform(test_y)




            # x = tf.reshape(x, shape=[batch_size, size])
            # x = tf.cast(x, dtype=tf.float32)
            train_accuracy = sess.run(accuracy, feed_dict={x_: test_x, y_: test_y, keep_prob: 1})
            lo = sess.run(loss, feed_dict={x_: x, y_: y, keep_prob: 1})
            # f1w = sess.run(fc1, feed_dict={x_: x, y_: y, keep_prob: 1})
            # f2w = sess.run(fc2, feed_dict={x_: x, y_: y, keep_prob: 1})
            # print("f1:",f1w,"\n","f2:",f2w)
            print("step %d, train accuracy %g , loss %g" % (i, train_accuracy, lo))
            # print(indexs)
        if i % 5000 == 0 and i != 0:
            save_path = os.path.join(log_path, "model.ckpt")
            saver.save(sess, save_path, global_step=i)

        sess.run(train_op, feed_dict={x_: x, y_: y, keep_prob: 0.5})


if __name__ == "__main__":
    train()
