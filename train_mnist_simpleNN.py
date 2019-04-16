# -*- coding:utf-8 -*-

from mnist import dense_to_one_hot

import tensorflow as tf
import numpy as np
import pandas as pd

#1 加载数据集
train = pd.read_csv('./dataset/train.csv')
#print(train)
images_train = train.iloc[:, 1:].values.astype(np.float) #所有行 从第2列到最后 第一列为标签

test = pd.read_csv('./dataset/test.csv')
images_test = test.iloc[:, :].values.astype(np.float)

#2 对输入数据进行处理，将其控制在0-1之间
images_train = np.multiply(images_train, 1.0 / 255)
iamges_test = np.multiply(images_test, 1.0 / 255)

#计算图片宽高
image_size = images_train.shape[1]  #列数
images_width = images_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

#3 处理标签，将标签结果取出来
labels_train = train.iloc[:, 0].values #标签
labels_count = np.unique(labels_train).shape[0]  #即一共有几种label，几分类问题

#对标签进行one-hot处理
labels = dense_to_one_hot(labels_train, labels_count).astype(np.uint8)

#4 对训练集进行划分
batch_size = 64
learning_rate = 0.1
epoches = 100
n_batch = len(images_train) // batch_size

#5 创建一个简单的神经网络用来对图片进行识别
x = tf.placeholder('float', shape=[None, image_size])
y = tf.placeholder('float', shape=[None, labels_count])


with tf.name_scope('input_reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, batch_size)


def weight_variable(shape):#初始化权重
    #正态分布,标准差为0.1
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):#初始化偏置
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

with tf.name_scope('weights'):
    weights = weight_variable([784, 10])
with tf.name_scope('baises'):
    biases = bias_variable([10])
with tf.name_scope('NN'):
    result = tf.matmul(x, weights) + biases
y_ = tf.nn.softmax(result)

#6 创建损失函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
tf.summary.scalar('loss', loss)
#7 优化
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#计算准确率
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

#合并
merged = tf.summary.merge_all()


#9 计算
with tf.Session() as sess:
    #初始化变量
    writer = tf.summary.FileWriter('./log/', sess.graph)
    sess.run(tf.global_variables_initializer())

    m_saver = tf.train.Saver()

    for epoch in range(1, epoches + 1):
        for batch in range(n_batch - 1):
            xs = images_train[batch * batch_size : (batch + 1) * batch_size]
            ys = labels[batch * batch_size : (batch + 1) * batch_size]
            #开始训练
            sess.run(train_step, feed_dict={x : xs, y : ys})
        xs = images_train[n_batch * batch_size : ]
        ys = labels[n_batch * batch_size : ]
        summary, _ = sess.run([merged, train_step], feed_dict={x : xs, y : ys})
        if epoch % 10 == 0:
            writer.add_summary(summary, epoch)
        out_loss, out_accuracy = sess.run([loss, accuracy], feed_dict={x : xs, y : ys})
        print("epoch%d: %f, accuracy: %.4f" % (epoch, out_loss, out_accuracy))
        if  epoch % 50 == 0:
            m_saver.save(sess, './model/mnist.ckpt', global_step=epoch)

    #预测
    myPrediction = sess.run(y_, feed_dict={x : images_test})
    sub = pd.DataFrame()
    sub['ImageId'] = np.arange(1, images_test.shape[0] + 1)
    sub['Label'] = np.argmax(myPrediction, axis=1)
    sub.to_csv("./log/prediction.csv", encoding='utf-8', index=False)

writer.close()


res = pd.read_csv("./log/prediction.csv")
print(res)


