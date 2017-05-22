
class InputData:
	def __init__(self)

def read_data(filename="data/ex1data1"):
	l = []
	with open(filename,"r") as f:
		line = f.readline()
		while line:		
			x,y = tuple(map(float,line.split(",")))
			l.append((x,y))
			line = f.readline()
	return l	

if __name__ == "__main__":
	print(read_data())
