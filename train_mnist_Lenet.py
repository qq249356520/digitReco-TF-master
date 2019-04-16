# -*- coding:utf-8 -*-

from mnist import dense_to_one_hot

import tensorflow as tf
import numpy as np
import pandas as pd
import csv

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
batch_size = 128
learning_rate = 1e-3
epoches = 100
n_batch = len(images_train) // batch_size

#5 创建一个简单的神经网络用来对图片进行识别
x = tf.placeholder('float', shape=[None, image_size])
y = tf.placeholder('float', shape=[None, labels_count])


def weight_variable(shape):#初始化权重
    #正态分布,标准差为0.1
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):#初始化偏置
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):
    #定义1 * 1卷积，不改变输入的shape
    return tf.nn.conv2d(x, w, [1, 1, 1, 1], padding='SAME')

def max_pool_22(x):
    #降维
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



with tf.name_scope('input_reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, batch_size)

with tf.name_scope('weights'):
    #计算32个特征,每个5*5 patch， #第三个参数是输入channel 第四个参数是输出channel
    #卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。
    weights_conv1 = weight_variable([5, 5, 1, 32])
    weights_conv2 = weight_variable([5, 5, 32, 64])
    weights_fc1 = weight_variable([7 * 7 * 64, 1024])
    weights_fc2 = weight_variable([1024, 10])
with tf.name_scope('baises'):
    biases_conv1 = bias_variable([32])
    biases_conv2 = bias_variable([64])
    biases_fc1 = bias_variable([1024])
    biases_fc2 = bias_variable([10])

with tf.name_scope('NN'):
    """
    现在我们可以开始实现了。每一层由一个卷积接一个max pooling完成。
    卷积在每个5x5的patch中算出out_channel个特征。
    而对于每一个输出通道都有一个对应的偏置量。
     """
    output = tf.nn.relu(conv2d(x_image, weights_conv1) + biases_conv1)
    output = max_pool_22(output) #14 × 14 * 32

    output = tf.nn.relu(conv2d(output, weights_conv2) + biases_conv2)
    output = max_pool_22(output)  #至此 图片变为了7 × 7 * 64

    #FC层 加入一个有1024个神经元的全连接层。并激活,将上一池化层生成的特征变为平铺向量
    output_flat = tf.reshape(output, [-1, 7*7*64])
    output = tf.nn.relu(tf.matmul(output_flat, weights_fc1) + biases_fc1)

    #为了减少过拟合，在输出层前加入dropout。
    #用一个placeholder代表一个神经元的输出在dropout中保持不变的概率。，这样便可在训练中启用dropout在测试中关闭。
    #tf的dropput不需要考虑神经元输出值的scale
    keep_prob = tf.placeholder(tf.float32)
    output = tf.nn.dropout(output, keep_prob)

    #输出层，10分类
    output = tf.nn.softmax(tf.matmul(output, weights_fc2) + biases_fc2)





#6 创建损失函数
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
tf.summary.scalar('loss', cross_entropy)
#7 优化
global_step = tf.Variable(0, name='global_step', trainable=False)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, global_step=global_step)

#计算准确率
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
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
            iter,_ = sess.run([global_step, train_step], feed_dict={x : xs, y : ys, keep_prob : 0.5})
            if(iter % 100 == 0):
                out_loss, out_accuracy = sess.run([cross_entropy, accuracy], feed_dict={x: xs, y: ys, keep_prob: 0.5})
                print("iter%d, epoch%d, loss: %f, accuracy: %.4f" % (iter, epoch, out_loss, out_accuracy))
        xs = images_train[n_batch * batch_size : ]
        ys = labels[n_batch * batch_size : ]
        summary, iter, _ = sess.run([merged, global_step, train_step], feed_dict={x : xs, y : ys, keep_prob : 0.5})
        if(iter % 100 == 0):
          out_loss, out_accuracy = sess.run([cross_entropy, accuracy], feed_dict={x: xs, y: ys, keep_prob: 0.5})
          print("iter%d, epoch%3d, loss: %f, accuracy: %.4f" % (iter, epoch, out_loss, out_accuracy))

        if epoch % 10 == 0:
            writer.add_summary(summary, epoch)

        if  epoch % 50 == 0:
            m_saver.save(sess, './model/mnist.ckpt', global_step=epoch)
writer.close()

#method 2
def savePred(prediction):
    #结果保存的路径
    with open(r'./log/prediction.csv', 'w') as myFile:
        myWriter = csv.writer(myFile)
        row = 0 #第一行的行号
        header = ['ImageId', 'Label']
        if row == 0:
            myWriter.writerow(header)
        for i in prediction:
            row += 1
            tmp = [row] #行号
            tmp.append(np.argmax(i)) #预测结果
#             myWriter.writerow(tmp)


with tf.Session() as sess1:
    #预测
    saver = tf.train.Saver()
    saver.restore(sess1, './model/mnist.ckpt-100')
    #一次预测28000内存不够，故分开预测
    test_batch_size = 4000
    test_n_batch = len(images_test) // test_batch_size
    allRes = []
    for batch in range(test_n_batch):
        test_batch_x = images_test[batch * test_batch_size : (batch + 1) * test_batch_size]
        myPrediction = sess1.run(output, feed_dict={x : test_batch_x, keep_prob : 1.0})
        allRes.extend(myPrediction)
        print(myPrediction)
    #method 1
    sub = pd.DataFrame()
    sub['ImageId'] = np.arange(1, images_test.shape[0] + 1)
    sub['Label'] = np.argmax(allRes, axis=1)
    sub.to_csv("./log/prediction.csv", index=False)










