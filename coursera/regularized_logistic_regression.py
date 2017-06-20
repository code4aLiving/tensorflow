
from __future__ import print_function
from utils import *
import pandas as pd
import tensorflow as tf

def plot_data(df):
	plotModel = PlotModel("Microchips", "Test 1", "Test 2")
	plot_logistic_regression_data(df, 0, 1, 2, plotModel)

def add_polinomial_features(df, degree):
	'''
	Add as columns in the data frame the polinomial fatures of the specified degree
	Ex. [x1,x2] => [1,x1,x2,x1^2,x1*x2,x2^2] for degree = 2
	'''
	pass

def logistic_regression(df):
	'''
	Logistic regression with the low level api
	Using regularization
	'''
	learning_rate = 0.01
	training_epochs = 1000
	batch_size = 10
	display_steps = 1
	
	#tf graph input
	x = tf.placeholder(tf.float32, [None, 28])
	y = tf.placeholder(tf.float32, [None, 1])

	#model weights
	W = tf.Variable(tf.zeros([28, 1]))
	b = tf.Variable(tf.zeros([1]))

	#Constructing the model
	pred = tf.nn.sigmoid(tf.matmul(x, -W))
	
	pass

if __name__ == "__main__":
	df = read_csv_data("data/ex2data2")
	plot_data(df)
	normalise(df, [0,1])
	plot_data(df)
