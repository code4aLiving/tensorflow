from utils import *
import matplotlib.pyplot as plt

if __name__=="__main__":
	df1 = read_csv_data("data/ex2data1")
#	print(df1)
#	plot_data(df1,0,1)
	df_positive = df1[df1[2] == 1]
	df_negative = df1[df1[2] == 0]
	plt.plot(df_positive[0], df_positive[1], 'go', df_negative[0], df_negative[1], 'rx')
	plt.title("Student admission data")
	plt.xlabel("Exam 1 score")
	plt.ylabel("Exam 2 score")
	plt.show()
