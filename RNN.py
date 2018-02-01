import numpy as np

class Network:
	def __init__(self,nh,nc):
		#Dimensions of Wxh,Whh,Why = [(nh,nc),(nh,nh),(nc,nh)]

		self.nc = nc
		self.nh = nh
		self.Wxh = np.random.randn(nh,nc) * np.sqrt(1/nc)
		self.Whh = np.random.randn(nh,nh) * np.sqrt(1/nh)
		self.Why = np.random.randn(nc,nh) * np.sqrt(1/nh)

		#Hidden state
		self.state = np.zeros((nh,1))

	def forward_pass(self,X,vocabulary):

		#Sequence is a list of one-hot vectors, each representing a character.
		#sequence = encode(X,vocabulary)
		#print(sequence)

		output = []	

		#len(X)+1 because we need a <start> token
		sequence = [0]*(len(X)+1)

		for i,xt in enumerate(X):
			sequence[i+1] = vocabulary.index(xt)

		#print("Sequence = ",sequence)

		for t in sequence:
			
			#Update the hidden state
			self.state = np.tanh(self.Wxh[:,t].reshape(self.nh,1) + np.dot(self.Whh,self.state))
			
			#Calculate Output
			yt = np.dot(self.Why,self.state)
			
			#print("shape of (state,yt) =({0},{1})".format(self.state.shape,yt.shape))			
			#Squash to provide a probability distribution between 0 and 1
			ot = softmax(yt)

			#print("ot =",ot)
			#prediction = np.argmax(ot,axis=1)
			output.append(ot)

		return output,sequence

def softmax(yt):
	exp_yt = np.exp(yt)
	return exp_yt/float(np.sum(exp_yt))

def predict(ot):
	#print("ot =",ot)
	#print("Finding argmax of",ot)
	return np.argmax(ot,axis=0)[0]

def load_data(filename):

	with open(filename,'r') as f:
		content = f.read()

	lines = list(filter(None,content.split('\n')))

	#print(lines)

	char_set = set()

	for line in lines:
		for char in line:
			char_set.add(char)

	#for i,char in enumerate(char_set):
		#print("{0}:{1}".format(i,char))

	return lines,list(char_set)

def calc_loss(output,target):

	# print(max(target))
	# print(len(output))
	#Output is a list of numpy arrays--> output after softmax
	#target is a list of integers --> integer equivalent of the character from the vocabulary
	loss = 0
	for i in range(len(output)):
		#print("i = ",i)
		#ot is a (nc,1) numpy array!!
		loss+= -np.log(output[i][target[i]][0])

	return loss/float(len(output))

def main():
	filename = "Immortals_of_Meluha.txt"
	#filename = "hello.txt"

	data,vocabulary = load_data(filename)

	vocabulary = ['<start>'] + vocabulary + ['<end>']

	print("Size of vocabulary = ",len(vocabulary))
	
	#print("vocabulary =",vocabulary)
	#for no,line in enumerate(data):
		#print("{0}:{1}".format(no,line))

	#data is a list of lines, each line is a string, i.e a list of characters
	my_rnn = Network(nh=100,nc=len(vocabulary))

	line = data[0]
	for line in data:

		output,sequence = my_rnn.forward_pass(line,vocabulary)

		#append the <end> token in target!
		sequence.append(len(vocabulary)-1)

		loss = calc_loss(output,sequence[1:])
		print("Loss = {}".format(loss))
		
		# print("Predictions Start here")
		# for char in output:
		# 	prediction = predict(char)
		# 	#print("Prediction =",pred)
		# 	#print(vocabulary[prediction],end="")
		# print("")

if __name__ == '__main__':
	main()