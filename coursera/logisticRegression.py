from utils import *
import matplotlib.pyplot as plt
import tensorflow as tf

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

	#estimator.fit(input_fn=input_fn_train)
	#estimator.predict(x=x)
	init = tf.global_variables_initializer()	
	with tf.Session() as sess:
		sess.run(init)
		sess.run(features[0])


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
	logistic_regression(df1)
