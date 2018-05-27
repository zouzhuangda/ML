# -*- coding: utf-8 -*-
# @Author: zouzh
# @Date:   2018-05-19 18:48:29
# @Last Modified by:   zouzh
# @Last Modified time: 2018-05-19 22:42:24
#2层简单神经网络（全连接）正向传播
import tensorflow as tf

#定义输入和参数
x = tf.constant([[0.7, 0.5]])
w1 = tf.Variable(tf.random_normal([2, 3], stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev = 1, seed = 1))

#定义前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#用会话计算结果
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	print(sess.run(w1))
	print(sess.run(w2))
	print(sess.run(a))
	print(sess.run(y))