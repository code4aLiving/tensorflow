import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_csv_data(filename):
	df = pd.read_csv(filename, header=None, delimiter=",", dtype=np.float64)
	return df

def plot_data(df, xcol, ycol):
	plt.plot(df[xcol], df[ycol], "rx")
	plt.show()	
