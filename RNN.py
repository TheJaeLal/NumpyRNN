import numpy as np

class Network:
	def __init__(self,nh,nc):
		#Dimensions of Wxh,Whh,Why = [(nh,nc),(nh,nh),(nc,nh)]

		self.Wxh = np.random.randn(nh,nc) * np.sqrt(1/nc)
		self.Whh = np.random.randn(nh,nh) * np.sqrt(1/nh)
		self.Why = np.random.randn(nc,nh) * np.sqrt(1/nh)

		#Hidden state
		self.state = np.zeros((nh,1))

	def forward_pass(self,data,vocabulary):

		#Size of vocabulary
		nc = len(vocabulary)
		
		for X in data:

			#Sequence is a list of one-hot vectors, each representing a character.
			sequence = encode(X)
				
			for xt in sequence:
				
				self.state = np.dot(self.Wxh,xt) + np.dot(self.Whh,self.state)
				yt = np.dot(self.Why,self.state)



		#Convert it into one-hot vector
		#Calculate the output
		#Take the softmax
		pass

def encode(X,nc):

	sequence = []
	
	#Append the <start> token to indicate start of the sequence/paragraph
	one_hot_start = np.zeros((nc,1))
	one_hot_start[0] = 1
	sequence.append(one_hot_start)

	for xt in X:
		char_idx = vocabulary.find(char)
		one_hot_xt = np.zeros((nc,1))
		one_hot_xt[char_idx] = 1
		sequence.append(one_hot_xt)
	
	#Append the <end> token to indicate end of the sequence/paragaraph
	one_hot_end = np.zeros((nc,1))
	one_hot_end[-1] = 1
	sequence.append(one_hot_end)

	return sequence

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
	filename = "Immortals_of_Meluha.txt"

	data,vocabulary = load_data(filename)

	vocabulary = ['<start>'] + vocabulary + ['<end>']

	#print(vocabulary)
	print(data)

	for no,line in enumerate(data):
		print("{0}:{1}".format(no,line))
	#data is a list of lines, each line is a string, i.e a list of characters
	#my_rnn = Network(nh=100,nc=len(vocabulary))

if __name__ == '__main__':
	main()