import numpy as np

class Network:
	def __init__(self,nh,nc):
		#Dimensions of Wxh,Whh,Why = [(nh,nc),(nh,nh),(nc,nh)]

		self.Wxh = np.random.randn(nh,nc) * np.sqrt(1/nc)
		self.Whh = np.random.randn(nh,nh) * np.sqrt(1/nh)
		self.Why = np.random.randn(nc,nh) * np.sqrt(1/nh)

	# def forward_pass(self,X):


def load_data(filename):

	#Breakdown text into Characters
	filename = "Immortals_of_Meluha.txt"
	#filename = "test.txt"

	with open(filename,'r') as f:
		content = f.read()

	lines = list(filter(None,content.split('\n')))

	# print(lines)

	char_set = set()

	for line in lines:
		for char in line:
			char_set.add(char)

	# for i,char in enumerate(char_set):
	# 	print("{0}:{1}".format(i,char))

	return lines,list(char_set)


def main():
	filename = "Immortals_of_Meluha"

	data,vocabulary = load_data(filename)

	my_rnn = Network(nh=100,nc=len(vocabulary))

if __name__ == '__main__':
	main()