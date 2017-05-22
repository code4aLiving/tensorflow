import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class Data:
	def __init__(self,x=[],y=[]):
		self.x = x
		self.y = y
	def get_x(self):
		return np.array(self.x,dtype=np.float)
	def get_y(self):
		return np.array(self.y,dtype=np.float)
	def plot(self):
		plt.plot(self.x,self.y,'ro')
		plt.show()

def read_data(filename="data/ex1data1"):
	data = Data()
	with open(filename,"r") as f:
		line = f.readline()
		while line:		
			x,y = tuple(map(float,line.split(",")))
			data.x.append(x)
			data.y.append(y)
			line = f.readline()
	return data

def linear_regression(data):
	tf.logging.set_verbosity(tf.logging.INFO)
	# Declare list of features. We only have one real-valued feature. There are many
	# other types of columns that are more complicated and useful.
	features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

	# An estimator is the front end to invoke training (fitting) and evaluation
	# (inference). There are many predefined types like linear regression,
	# logistic regression, linear classification, logistic classification, and
	# many neural network classifiers and regressors. The following code
	# provides an estimator that does linear regression.
	estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

	# TensorFlow provides many helper methods to read and set up data sets.
	# Here we use `numpy_input_fn`. We have to tell the function how many batches
	# of data (num_epochs) we want and how big each batch should be.
	x = data.get_x()
	y = data.get_y()
#	x = np.array([1., 2., 3., 4.])
#	y = np.array([0., -1., -2., -3.])
	input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4, num_epochs=1000)

	# We can invoke 1000 training steps by invoking the `fit` method and passing the
	# training data set.
	estimator.fit(input_fn=input_fn, steps=1000)

	# Here we evaluate how well our model did. In a real example, we would want
	# to use a separate validation and testing data set to avoid overfitting.
	print(estimator.evaluate(input_fn=input_fn))
	variables = estimator.get_variable_names()
	for v in variables:
		print(v,estimator.get_variable_value(v))
	x_p = np.array([1.,2.])
	i = 0
	for p in estimator.predict({"x":x_p},as_iterable=True):
		print(x_p[i],p)
		i+=1

if __name__ == "__main__":
	data = read_data()
#	print(len(data.x),len(data.y))
	
	#data.plot()
	linear_regression(data)
