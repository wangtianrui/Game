import tensorflow as tf
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import os
from sklearn import preprocessing
import copy
import f2_score

num_class = 2
batch_size = 15000
train_path = r"train_ALL.csv"

learning_rate = 0.00000001
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
size = 62
log_path = r"./log2/"


def readData(path):
    trainData = pd.read_csv(path)
    trainData['emp_length'].fillna(trainData['emp_length'].median(), inplace=True)
    trainData['annual_inc'].fillna(trainData['annual_inc'].median(), inplace=True)
    trainData['title'].fillna(trainData['title'].median(), inplace=True)
    trainData['pub_rec'].fillna(trainData['pub_rec'].median(), inplace=True)
    trainData['revol_util'].fillna(trainData['revol_util'].median(), inplace=True)
    trainData['total_acc'].fillna(trainData['total_acc'].median(), inplace=True)
    trainData['collections_12_mths_ex_med'].fillna(trainData['collections_12_mths_ex_med'].median(), inplace=True)
    whole_list = np.array(trainData, "float")

    return whole_list


global_maker_flag = 1


def makeData(copy_data, flag):
    # copy_data = copy.deepcopy(data)
    # print("1:",len(data[0]))
    # global global_maker_flag
    # print(len(data))
    # print(len(data[1480]))
    # print(data)
    if flag == "train":
        all_data = random.sample(list(copy_data), batch_size)
        y = [[temp[45]] for temp in all_data]
        x = np.delete(all_data, 45, axis=1)
        # print("test:", len(data[0]))
        # global_maker_flag += 1
        x = [list(temp) for temp in x]
    else:
        all_data = random.sample(list(copy_data), 10000)
        y = [[temp[45]] for temp in all_data]
        x = np.delete(all_data, 45, axis=1)
        # print("test:", len(data[0]))
        # global_maker_flag += 1
        x = [list(temp) for temp in x]
    # print(label)
    # print("lenx:",len(x))
    # print("len x1:",len(x[1]))
    # x = tf.reshape(x, shape=[batch_size, size])
    # label = tf.reshape(label, shape=[batch_size, 2])
    return x, y


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

    return fc3


def fc_2(input, keep_prob):
    fc1_w = getWeight([size, 128])
    fc1_b = getBias([128])
    fc1 = tf.nn.relu(tf.matmul(input, fc1_w) + fc1_b)

    h_fc1_drop = tf.nn.dropout(fc1, keep_prob)
    fc3_w = getWeight([128, 2])
    fc3_b = getBias([2])
    fc3 = tf.matmul(h_fc1_drop, fc3_w) + fc3_b

    return fc3


def cross_entropy(fc_result, label):
    label = tf.one_hot(label, depth=num_class)
    return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=fc_result, labels=label))


def train():
    x_ = tf.placeholder(shape=[batch_size, size], dtype="float")
    y_ = tf.placeholder(shape=[batch_size, 1], dtype="int32")
    keep_prob = tf.placeholder("float")

    x_test = tf.placeholder(shape=[10000, size], dtype="float")
    y_test = tf.placeholder(shape=[10000, 1], dtype="int32")

    global_step = tf.Variable(0, name="global_step", trainable=False)
    logits = fc_2(x_, keep_prob)
    logits_test = fc(x_test, keep_prob)
    loss = cross_entropy(logits, y_)
    # correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tp_op, tn_op, fp_op, fn_op = f2_score.get_f2_score(label=y_test, logits=logits_test)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    sess.run(init)
    saver = tf.train.Saver(tf.global_variables())
    whole_data = readData(train_path)
    test_data = readData(train_path)
    # test_data11 = readData(test_path11)
    # whole_data = sess.run(read_data)
    for i in range(200000):
        x, y = makeData(whole_data, "train")
        test_x, test_y = makeData(test_data, "train")
        min_max_scaler = preprocessing.MinMaxScaler()
        x = min_max_scaler.fit_transform(x)

        # 标准化
        ss_x = StandardScaler()
        ss_y = StandardScaler()
        x = ss_x.fit_transform(x)
        y = ss_y.fit_transform(y)

        if i % 200 == 0 and i != 0:
            save_path = os.path.join(log_path, "model.ckpt")
            saver.save(sess, save_path, global_step=i)

        _, lo = sess.run([train_op, loss], feed_dict={x_: x, y_: y, keep_prob: 0.5})
        print("step %d, loss %g" % (i, lo))


if __name__ == "__main__":
    train()
