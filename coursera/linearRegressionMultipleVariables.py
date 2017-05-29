
import pandas as pd
import numpy as np
import tensorflow as tf
import random as rd
import matplotlib.pyplot as plt

def generate_fake_data():
	df = pd.DataFrame(columns=[0,1,2])
	for x in range(100):
		x1 = float(rd.randint(1000,5*10**5))
		x2 = float(rd.randint(1,5))
		df2 = pd.DataFrame([[x1,x2,(0.5*x1)+(2*x2)+1]])
		df = df.append(df2, ignore_index=True)
	return df

def read_csv_data(filename):
	df = pd.read_csv(filename, header=None, delimiter=",",dtype=np.float64)
	return df

def plot_data(df,col):
	y = df.columns[-1]
	plt.plot(df[col],df[y],'ro')
	plt.show()

def normalize(df):
	res = df.copy()
	for c in df.columns:
		mean = df[c].mean()
		std = df[c].std()
		res[c] = (res[c]-mean)/std
	return res

def linear_regression(df):
	print(df)
	print(np.array(df[0].values))
	tf.logging.set_verbosity(tf.logging.INFO)
	# Declare list of features. We only have one real-valued feature. There are many
	# other types of columns that are more complicated and useful.
	features = []
	input_dict = {}
	col_mean_std = {}
	for c in df.columns[:-1]:
		mean = df[c].mean()
		std = df[c].std()
		col_mean_std[c] = (mean,std)
		features.append(tf.contrib.layers.real_valued_column("x"+str(c), dimension=1,
							normalizer=lambda x:(x-mean)/std))
		input_dict["x"+str(c)]=np.array(df[c].values)

	# An estimator is the front end to invoke training (fitting) and evaluation
	# (inference). There are many predefined types like linear regression,
	# logistic regression, linear classification, logistic classification, and
	# many neural network classifiers and regressors. The following code
	# provides an estimator that does linear regression.
	estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

	# TensorFlow provides many helper methods to read and set up data sets.
	# Here we use `numpy_input_fn`. We have to tell the function how many batches
	# of data (num_epochs) we want and how big each batch should be.
#	x1 = np.array(df[0].values)
#	x2 = np.array(df[1].values)
	mean = df[len(df.columns)-1].mean()
	std = df[len(df.columns)-1].std()
	df[len(df.columns)-1] = (df[len(df.columns)-1]-mean)/std
	y = np.array(df[len(df.columns)-1].values)
	col_mean_std[len(df.columns)-1]=(mean,std)
	input_fn = tf.contrib.learn.io.numpy_input_fn(input_dict, y, batch_size=4, num_epochs=1000)

	# We can invoke 1000 training steps by invoking the `fit` method and passing the
	# training data set.
	estimator.fit(input_fn=input_fn, steps=1000)

	# Here we evaluate how well our model did. In a real example, we would want
	# to use a separate validation and testing data set to avoid overfitting.
	print(estimator.evaluate(input_fn=input_fn))
	
	variables = estimator.get_variable_names()
	for v in variables:
		print(v,estimator.get_variable_value(v))
	return estimator,col_mean_std


if __name__ == "__main__":
	filename = "data/ex1data2"
	df = read_csv_data(filename)
	dfp = normalize(df)
	plot_data(dfp,0)
	plot_data(dfp,1)

#	df = generate_fake_data()
	estimator, col_mean_std = linear_regression(df)
	x1 = np.array([(1600-col_mean_std[0][0])/col_mean_std[0][1]])
	x2 = np.array([(3-col_mean_std[1][0])/col_mean_std[1][1]])
#	print("Evaluating --------------")
	for p in estimator.predict({"x0":x1,"x1":x2},as_iterable=True):
		print(x1,x2,p*col_mean_std[2][1]+col_mean_std[2][0])
