# -*- coding:utf-8 -*-
"""
    对所给mnist数据集进行预处理
"""

import numpy as np

#one-hot编码,将label变为二维数组，当前标签数字置1
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes #结果为 [     0     10     20 ... 419970 419980 419990] 每个10 相当于代表了一次分类 一共num_labels个

    labels_one_hot = np.zeros((num_labels, num_classes))
    #flatten和ravel都是平铺，区别是一个返回的是视图（改变元素值影响原数组），一个是拷贝（不影响）
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1 #flat 返回的是一个迭代器，可用索引对其引导，作用是把一个array看做平铺
    return labels_one_hot

