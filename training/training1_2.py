# -*- coding: utf-8 -*-
# @Author: zouzh
# @Date:   2018-05-19 22:33:55
# @Last Modified by:   zouzh
# @Last Modified time: 2018-05-19 22:50:55
#反向传播
import tensorflow as tf

#准备数据
x = tf.constant([[0.7, 0.5]])
w1 = tf.Variable(tf.random_normal([2, 4]))
w2 = tf.Variable(tf.random_normal([4, 3]))
w3 = tf.Variable(tf.random_normal([3, 1]))

#构建向前传播过程
a1 = tf.matmul(x, w1)
a2 = tf.matmul(a1, w2)
y = tf.matmul(a2, w3)

#计算结果
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	print(sess.run(w1))
	print(sess.run(w2))
	print(sess.run(w3))
	print(sess.run(a1))
	print(sess.run(a2))
	print(sess.run(y))