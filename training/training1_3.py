import tensorflow as tf
import numpy as np
from itertools import islice

# 准备数据
BATH_SIZE = 10 #一次喂入神经网络的数据尺寸

# filename = "C:\\Users\\joe\\Documents\\iris\\iris.data"
filename = "../ExerciseData/iris_training.csv"
filename_test = "../ExerciseData/iris_test.csv"

data = None
data_test = None
with open(filename) as file:
	for line in islice(file, 0, None):
		data = file.read().split()


with open(filename_test) as file:
	for line in islice(file, 0, None):
		data_test = file.read().split()

fetures = np.zeros((len(data),4))
lables = np.zeros((len(data),1))

fetures_test = np.zeros((len(data_test),4))

# def f(iris_name):
# 	if iris_name == 'Iris-setosa':
# 		return 1.
# 	elif iris_name == 'Iris-versicolor':
# 		return 2.
# 	elif iris_name == 'Iris-virginica':
# 		return 3.

for i in range(len(data)):
	line = data[i].split(',')
	# fetures.append(np.array(line[0:4],dtype = np.float32))
	# fetures[i] = ([float(line[0]),float(line[1]),float(line[2]),float(line[3])])
	# lables.append([f(line[4])])
	fetures[i] = line[0:4]
	# lables[i] = [f(line[4])]
	lables[i] = [line[4]]

for i in range(len(data_test)):
	line = data_test[i].split(',')
	fetures_test[i] = line[0:4]

#构建计正向传播数据图
x = tf.placeholder(tf.float32, shape = (None, 4))
y_ = tf.placeholder(tf.float32, shape = (None, 1))

w1 = tf.Variable(tf.random_normal([4, 4]))
w2 = tf.Variable(tf.random_normal([4, 1]))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#滑动学习率
LEARNING_RATE_BASE = 0.002 #最初学习率
LEARNING_RATE_DECAY = 0.90 #学习率衰减率
LEARNING_RATE_STEP = 1 #喂入STEP轮BATCH_SIZE后，更新一次学习率，一般设为：总样本数/BATCH_SIZE

#运行了几轮BATH_SIZE的计数器，初始值0，trainable=Flase表示不作为变量进行训练
global_setp = tf.Variable(0,trainable=False)
#定义指数下降学习率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_setp,LEARNING_RATE_STEP,LEARNING_RATE_DECAY,staircase=True)


#反向传播
loss = tf.reduce_mean(tf.square(y - y_))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
# train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#训练
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)

	for i in range(30000):
		start = (i * BATH_SIZE) % 120
		end = start + BATH_SIZE
		sess.run(train_step, feed_dict = {x:fetures[start:end], y_:lables[start:end]})
		
		if i % 3000 == 0:
			total_loss = sess.run(loss, feed_dict = {x:fetures,y_:lables})
			print('step:',i,
				# '\nw1:',sess.run(w1),
				# '\nw2:',sess.run(w2),
				'\nloss:',total_loss)
	print('\n')
	print('w1:\n', sess.run(w1))
	print('w2:\n', sess.run(w2))
	print('loss:\n', sess.run(loss, feed_dict = {x:fetures,y_:lables}))
	print('test\n',sess.run(y, feed_dict = {x:fetures_test}))