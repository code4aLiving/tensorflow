

import pandas as pd
import numpy as np
import tensorflow as tf
import random as rd

def generate_fake_data():
	df = pd.DataFrame(columns=[0,1,2])
	for x in range(100):
		x1 = float(rd.randint(1,50))
		x2 = float(rd.randint(1,50))
		df2 = pd.DataFrame([[x1,x2,0.5*x1+2*x2+3]])
		df = df.append(df2, ignore_index=True)
	return df

def read_csv_data(filename):
	df = pd.read_csv(filename, header=None, delimiter=",")
	return df

def normalise(df,col_index):
	#remove possible duplicates
	col_index = list(set(col_index))

	resdf = df.copy()
	col_mean_std = {}
	for c in col_index:
		col_mean_std[c]=(resdf[c].mean(),resdf[c].std())
		resdf[c] = (resdf[c]-resdf[c].mean())/resdf[c].std()
	return resdf,col_mean_std

def linear_regression(df):
	#print(df)
	tf.logging.set_verbosity(tf.logging.INFO)
	# Declare list of features. We only have one real-valued feature. There are many
	# other types of columns that are more complicated and useful.
	features = [tf.contrib.layers.real_valued_column("x1", dimension=1),
		    tf.contrib.layers.real_valued_column("x2",dimension=1)]

	# An estimator is the front end to invoke training (fitting) and evaluation
	# (inference). There are many predefined types like linear regression,
	# logistic regression, linear classification, logistic classification, and
	# many neural network classifiers and regressors. The following code
	# provides an estimator that does linear regression.
	estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

	# TensorFlow provides many helper methods to read and set up data sets.
	# Here we use `numpy_input_fn`. We have to tell the function how many batches
	# of data (num_epochs) we want and how big each batch should be.
	x1 = df[[0]].values
	x2 = df[[1]].values
	y = df[[2]].values
	
	#print(x1)
	#print(x2)
	#print(y)	
#	x = np.array([1., 2., 3., 4.])
#	y = np.array([0., -1., -2., -3.])
	input_fn = tf.contrib.learn.io.numpy_input_fn({"x1":x1,"x2":x2}, y, batch_size=4, num_epochs=1000)

	# We can invoke 1000 training steps by invoking the `fit` method and passing the
	# training data set.
	estimator.fit(input_fn=input_fn, steps=1000)

	# Here we evaluate how well our model did. In a real example, we would want
	# to use a separate validation and testing data set to avoid overfitting.
	print(estimator.evaluate(input_fn=input_fn))
	
	variables = estimator.get_variable_names()
	for v in variables:
		print(v,estimator.get_variable_value(v))
	return estimator


if __name__ == "__main__":
	filename = "data/ex1data2"
#	df = read_csv_data(filename)
	df = generate_fake_data()
	#print(df)
	#df,col_mean_std = normalise(df,[0,1])
#	print(col_mean_std)
#	print(df)
	estimator = linear_regression(df)
	#x1 = np.array([(3-col_mean_std[0][0])/col_mean_std[0][1]])
	#x2 = np.array([(3-col_mean_std[1][0])/col_mean_std[1][1]])
#	print("Evaluating --------------")
	#for p in estimator.predict({"x1":x1,"x2":x2},as_iterable=True):
#		print(x1,x2,p)

