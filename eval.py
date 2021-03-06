import tensorflow as tf
import numpy as np
import csv
import input_data
import the_final_trian
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from sklearn import preprocessing

size = 35
log_path = "./log2"
test_path = "./data/wang_test_id.csv"


def readData(path):
    whole_data = pd.read_csv(path, encoding='utf-8')
    whole_data = whole_data.fillna(0)
    # print(whole_data)
    whole_list = np.array(whole_data)
    for i in range(len(whole_list)):
        for temp in range(len(whole_list[i])):
            if whole_list[i][temp] == " ":
                whole_list[i][temp] = 0
    # print("转换前：", whole_list)
    whole_list = np.array(whole_list, dtype="float")
    # print(whole_list)
    whole_list = np.delete(whole_list, [6, 14], axis=1)
    print(len(whole_list))
    print(len(whole_list[1]))
    return whole_list


def makeData(data):
    x = []
    ids = []
    for index in range(len(data)):
        row = data[index]
        id = int(row[35])
        x.append(row[:35])
        ids.append(id)
    return x , ids


def eval_item(x):
    logits, _, _ = the_final_trian.fc_2(x, 1)
    y = tf.argmax(logits, 1)
    return y


count = 0

if __name__ == "__main__":
    labels = []
    # [177476 rows x 37 columns]
    x_ = tf.placeholder(shape=[1, size], dtype="float")
    # print(x_)
    y_ = eval_item(x_)
    ckpt = tf.train.get_checkpoint_state(log_path)
    whole_list = readData(test_path)
    x , ids= makeData(whole_list)
    # print(len(x[1]))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        ss_x = StandardScaler()
        min_max_scaler = preprocessing.MinMaxScaler()
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord)  # 启动QueueRunner, 此时文件名队列已经进队。
        ckpt = tf.train.get_checkpoint_state(log_path)
        i = 0
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            try:
                while not coord.should_stop() and i < 1:
                    # image = sess.run(image_batch)
                    # print(image)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    print("模型为:", global_step)
                    x = ss_x.fit_transform(x)
                    x = min_max_scaler.fit_transform(x)
                    print(len(x))
                    for p in range(len(x)):
                        temp = x[p]
                        temp = [temp]
                        id = ids[p]
                        y = sess.run(y_, feed_dict={x_: temp})
                        # plot_images(image_batch, label_batch)
                        # print(y[0])
                        labels.append([id,y[0]])
                        count += 1
                        print(count)
                        # print(labels)
                        # print(labels)
                    i += 1
            except tf.errors.OutOfRangeError:
                print('done!')
            finally:
                coord.request_stop()
            coord.join(threads)
            sess.close()

        with open(r"./data/01exchange.csv", "w", newline="") as f:
            # for i in database:
            #     for temp in database:
            #         f.write(temp)
            #         f.write("\t")
            #     f.write("\n")
            writer = csv.writer(f)
            # count = 0
            for i in labels:
                print(i)
                if i[1] == 0:
                    i[1] = 1
                    writer.writerow(i)
                elif i[1] == 1:
                    i[1] = 0
                    writer.writerow(i)
                else:
                    print(i[1])

