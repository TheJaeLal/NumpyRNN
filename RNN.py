import numpy as np

class Network:
	def __init__(self,nh,nc):
		#Dimensions of Wxh,Whh,Why = [(nh,nc),(nh,nh),(nc,nh)]

		self.Wxh = np.random.randn(nh,nc) * np.sqrt(1/nc)
		self.Whh = np.random.randn(nh,nh) * np.sqrt(1/nh)
		self.Why = np.random.randn(nc,nh) * np.sqrt(1/nh)

		#Hidden state
		self.state = np.zeros((nh,1))

	def forward_pass(self,X,vocabulary):

		#Sequence is a list of one-hot vectors, each representing a character.
		sequence = encode(X,vocabulary)
		
		output = []	
		for xt in sequence:
			
			#Update the hidden state
			self.state = np.dot(self.Wxh,xt) + np.dot(self.Whh,self.state)
			
			#Calculate Output
			yt = np.dot(self.Why,self.state)
			
			#Squash to provide a probability distribution between 0 and 1
			ot = softmax(yt)

			#prediction = np.argmax(ot,axis=1)
			output.append(ot)

		return output

def softmax(yt):
	exp_yt = np.exp(yt)
	return exp_yt/float(np.sum(exp_yt))

def predict(ot):
	return np.argmax(ot,axis=0)

def encode(X,vocabulary):

	nc = len(vocabulary)

	sequence = []
	
	#Append the <start> token to indicate start of the sequence/paragraph
	one_hot_start = np.zeros((nc,1))
	one_hot_start[0] = 1
	sequence.append(one_hot_start)

	for xt in X:
		char_idx = vocabulary.index(xt)
		one_hot_xt = np.zeros((nc,1))
		one_hot_xt[char_idx] = 1
		sequence.append(one_hot_xt)
	
	#Append the <end> token to indicate end of the sequence/paragaraph
	one_hot_end = np.zeros((nc,1))
	one_hot_end[-1] = 1
	sequence.append(one_hot_end)

	return sequence

def load_data(filename):

	with open(filename,'r') as f:
		content = f.read()

	lines = list(filter(None,content.split('\n')))

	print(lines)

	char_set = set()

	for line in lines:
		for char in line:
			char_set.add(char)

	for i,char in enumerate(char_set):
		print("{0}:{1}".format(i,char))

	return lines,list(char_set)


def main():
	#filename = "Immortals_of_Meluha.txt"
	filename = "hello.txt"

	data,vocabulary = load_data(filename)

	vocabulary = ['<start>'] + vocabulary + ['<end>']

	# for no,line in enumerate(data):
	# 	print("{0}:{1}".format(no,line))

	#data is a list of lines, each line is a string, i.e a list of characters
	my_rnn = Network(nh=100,nc=len(vocabulary))

	line = data[0]
	output = my_rnn.forward_pass(line,vocabulary)
	print(output)
	print("Predictions")
	for char in output:
		print(vocabulary[int(predict(char))])

if __name__ == '__main__':
	main()