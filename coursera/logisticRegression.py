from __future__ import print_function
from utils import *
import matplotlib.pyplot as plt
import tensorflow as tf
import math

def log_reg(df):
	
	learning_rate = 0.01
	training_epochs = 1000
	batch_size = 10
	display_step = 1
	
	#tf Graph input
	x = tf.placeholder(tf.float32, [None, 2])
	y = tf.placeholder(tf.float32, [None, 1])
	
	#set model weights
	W = tf.Variable(tf.zeros([2, 1]))
	b = tf.Variable(tf.zeros([1]))

	#Construct model
#	pred = tf.nn.softmax(tf.matmul(x, W) + b) #Softmax
	pred = tf.nn.sigmoid(tf.matmul(x, -W))
	# Minimize error using cross entropy
#	cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=0))
	cost = tf.reduce_mean(tf.reduce_sum(-y * tf.log(pred) - (1 - y) * tf.log(1 - pred)))
	# Gradient descent
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	#Initializing the variables
	init = tf.global_variables_initializer()
	
	#Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		
		#training cycle
		for epoch in range(training_epochs):
			avg_cost = 0
			total_batch = int(len(df)/batch_size)
			for i in range(total_batch):
				batch_df = df[i*batch_size:(i+1)*batch_size]
				batch_x = batch_df[[0,1]]
				batch_y = batch_df[[2]]
		#		print(batch_x)
		#		print(batch_y)
				_,c,p,yy,ww = sess.run([optimizer, cost, pred,y,W], feed_dict={x: batch_x, y: batch_y})
			#	print("Prediction : {} , Cost : {}".format(p,c))
			#	print("W:{}".format(ww))
			#	print("-----------")
				#Compute average loss
				avg_cost += c/total_batch

			#display logs per epoch step
			if not (epoch + 1) % display_step:
				print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
		#_,c,p,yy = sess.run([optimizer, cost, pred, y], feed_dict={x: df[[0,1]], y: df[[2]]})
		#print("Optimization finished")
		#print(p,yy)
		#print(len(p),len(yy))	
		# Test model
		pred_binary = tf.cast(tf.greater(pred, 0.5), tf.float32)
		correct_prediction = tf.equal(pred_binary, y)
		# Calculate accuracy
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		print("Accuracy:", accuracy.eval({x:df[[0,1]] , y:df[[2]]}))

	

def logistic_regression(df):
	target = df.columns[-1]
	print(df[target])
	# See tf.contrib.learn.Estimator(...) for details on model_fn structure
	def my_model_fn(features, labels, mode):
    		pass
	features = tf.estimator.inputs.pandas_input_fn(
			   x=df, target_column=target, shuffle=False)()
	print(features)
	print(len(features))
	estimator = tf.contrib.learn.LogisticRegressor(model_fn=my_model_fn)

	# Input builders
	def input_fn_train():
		pass

	estimator.fit(input_fn=input_fn_train)
	estimator.predict(x=x)
	init = tf.global_variables_initializer()	
	#with tf.Session() as sess:
	#	sess.run(init)
	#	sess.run(features[0])


def plot_logistic_regression_data(df):
	df_positive = df1[df1[2] == 1]
	df_negative = df1[df1[2] == 0]
	plt.plot(df_positive[0], df_positive[1], 'go', df_negative[0], df_negative[1], 'rx')
	plt.title("Student admission data")
	plt.xlabel("Exam 1 score")
	plt.ylabel("Exam 2 score")
	plt.show()

if __name__=="__main__":
	df1 = read_csv_data("data/ex2data1")
	for i in range(len(df1.columns)-1):
		df1[[i]] = (df1[[i]] - df1[[i]].mean()) / df1[[i]].std()
	
	print(len(df1))
#	print(df1)
	#print(df1)
	#print(df1[[0,1]])
	#print(df1[[2]])
	#logistic_regression(df1)
	log_reg(df1)
