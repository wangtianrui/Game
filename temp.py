import tensorflow as tf
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import os
from sklearn import preprocessing

num_class = 2
batch_size = 100
train_path = r"wang_data.csv"
test_path = r"wang_data_test5500.csv"
test_path11 = r"wang_data_test200.csv"
learning_rate = 0.00000001
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
size = 35
log_path = r"./log2/"


def get_f2_score(label, logits):
    predictions = tf.argmax(logits, 1)
    actuals = tf.argmax(label, 1)

    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)

    tp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    tn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )

    fp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    fn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )

    return tp_op, tn_op, fp_op, fn_op


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
    whole_list = np.delete(whole_list, [6, 14], axis=1)
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
            label = [int(row[size])]
            y.append(label)
            row = np.delete(row, size)
            x.append(row)
            ran.append(index)
    elif flag == "11":
        for i in range(batch_size):
            index = random.randint(0, 190)
            # print("test data len", len(data))
            row = data[index]
            label = [int(row[size])]
            y.append(label)
            row = np.delete(row, size)
            x.append(row)
            ran.append(index)
    else:
        for i in range(batch_size):
            index = random.randint(0, 5500)
            # print("test data len", len(data))
            row = data[index]
            label = [int(row[size])]
            y.append(label)
            row = np.delete(row, size)
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

    fc2_w = getWeight([128, 512])
    fc2_b = getBias([512])
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_w) + fc2_b)

    h_fc1_drop = tf.nn.dropout(fc2, keep_prob)
    fc3_w = getWeight([512, 2])
    fc3_b = getBias([2])
    fc3 = tf.matmul(h_fc1_drop, fc3_w) + fc3_b

    return fc3, fc1_w, fc2_w


def fc_2(input, keep_prob):
    fc1_w = getWeight([size, 128])
    fc1_b = getBias([128])
    fc1 = tf.nn.relu(tf.matmul(input, fc1_w) + fc1_b)

    h_fc1_drop = tf.nn.dropout(fc1, keep_prob)
    fc3_w = getWeight([128, 2])
    fc3_b = getBias([2])
    fc3 = tf.matmul(h_fc1_drop, fc3_w) + fc3_b

    return fc3, 1, 1


def cross_entropy(fc_result, label):
    label = tf.one_hot(label, depth=num_class)
    return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=fc_result, labels=label))



def train():
    x_ = tf.placeholder(shape=[batch_size, size], dtype="float")
    y_ = tf.placeholder(shape=[batch_size, 1], dtype="int32")
    keep_prob = tf.placeholder("float")

    global_step = tf.Variable(0, name="global_step", trainable=False)
    logits, fc1, fc2 = fc_2(x_, keep_prob)
    loss = cross_entropy(logits, y_)
    # print("logits:",logits)

    # correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    tp_op, tn_op, fp_op, fn_op = get_f2_score(label=y_, logits=logits)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    init = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    sess.run(init)
    saver = tf.train.Saver(tf.global_variables())
    whole_data = readData(train_path)
    test_data = readData(test_path)
    test_data11 = readData(test_path11)
    # whole_data = sess.run(read_data)
    for i in range(3000000):
        if i % 100 == 0:
            index = 0
            x, y, indexs = makeData(whole_data, "train")
            if random.randint(0, 10) < 5:
                index = 5000
                test_x, test_y, test_index = makeData(test_data, "5000")
            else:
                index = 200
                test_x, test_y, test_index = makeData(test_data11, "11")

            # 标准化
            ss_x = StandardScaler()
            ss_y = StandardScaler()
            x = ss_x.fit_transform(x)
            y = ss_y.fit_transform(y)
            # print("x:", x)


            test_x = ss_x.transform(test_x)
            # print("test x:", x)
            test_y = ss_y.transform(test_y)

            # 归一化
            min_max_scaler = preprocessing.MinMaxScaler()
            x = min_max_scaler.fit_transform(x)
            test_x = min_max_scaler.transform(test_x)

            # x = tf.reshape(x, shape=[batch_size, size])
            # x = tf.cast(x, dtype=tf.float32)
            tp, tn, fp, fn = sess.run([tn_op, tp_op, fp_op, fn_op], feed_dict={x_: test_x, y_: test_y, keep_prob: 1})
            try:
                tpr = float(tp) / (float(tp) + float(fn))
                fpr = float(fp) / (float(tp) + float(fn))

                accuracy = (float(tp) + float(tn)) / (float(tp) + float(fp) + float(fn) + float(tn))
            except ZeroDivisionError:
                accuracy = 0.0
            recall = tpr
            precision = float(tp) / (float(tp) + float(fp))
            f2 = ((2 ** 2 + 1) * (precision * recall)) / ((2 ** 2) * precision + recall)
            lo = sess.run(loss, feed_dict={x_: x, y_: y, keep_prob: 1})
            # f1w = sess.run(fc1, feed_dict={x_: x, y_: y, keep_prob: 1})
            # f2w = sess.run(fc2, feed_dict={x_: x, y_: y, keep_prob: 1})
            # print("f1:",f1w,"\n","f2:",f2w)
            print("step %d, f2_score %g ,Accuracy %g , loss %g , this testdataset:%d" % (i, f2, accuracy, lo, index))
            # print(indexs)
        if i % 5000 == 0 and i != 0:
            save_path = os.path.join(log_path, "model.ckpt")
            saver.save(sess, save_path, global_step=i)

        sess.run(train_op, feed_dict={x_: x, y_: y, keep_prob: 0.5})


if __name__ == "__main__":
    train()