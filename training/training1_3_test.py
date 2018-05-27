import tensorflow as tf

savepath = "C:\\Users\\joe\\OneDrive\\Documents\\ML\\ExerciseData\\save"
filename_test = "C:\\Users\\joe\\OneDrive\\Documents\\ML\\ExerciseData\\iris_test.csv"

with tf.Session() as sess:
	save = tf.train.Saver()
	save.restore(sess,savepath)
	sess.run(y,sess)
