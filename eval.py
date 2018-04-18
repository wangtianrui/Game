import tensorflow as tf
import numpy as np
import train_lihao
import cv2

test_batch_size = 1
num_class = 2

CHECKPOINT = "log"  #


def inference(input):  # 1 x 24 x 24 x 3
    logits = train_lihao.fc(input, 1)
    results = tf.arg_max(logits, 1)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord)  # 启动QueueRunner, 此时文件名队列已经进队。
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT)
        i = 0
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            try:
                while not coord.should_stop() and i < 1:
                    # image = sess.run(image_batch)
                    # print(image)
                    for i in range(6):
                        i = sess.run(ima)
                        result = sess.run(results, feed_dict={image: i})
                        print("预测结果是", result)
                        i += 1
            except tf.errors.OutOfRangeError:
                print('done!')
            finally:
                coord.request_stop()
            coord.join(threads)
            sess.close()
