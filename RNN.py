import numpy as np

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

def calc_cost(output,target):

	# print(max(target))
	# print(len(output))
	#Output is a list of numpy arrays--> output after softmax
	#target is a list of integers --> integer equivalent of the character from the vocabulary
	cost = 0
	for i in range(len(output)):
		#print("i = ",i)
		#ot is a (nc,1) numpy array!!
		cost+= -np.log(output[i][target[i],0])

	return cost/float(len(output))

def softmax(yt):
	exp_yt = np.exp(yt)
	return exp_yt/float(np.sum(exp_yt))

def predict(ot):
	#print("ot =",ot)
	#print("Finding argmax of",ot)
	return np.argmax(ot,axis=0)[0]

class Network:
	def __init__(self,nh,nc):
		#Dimensions of Wx,Wh,Wy = [(nh,nc),(nh,nh),(nc,nh)]

		self.nc = nc
		self.nh = nh
		#Weights for input layer
		self.Wx = np.random.randn(nh,nc) * np.sqrt(1/nc)
		
		#Weights for hidden layer(Recursive weights, from one hidden state to another)
		self.Wh = np.random.randn(nh,nh) * np.sqrt(1/nh)

		#Output Weights
		self.Wy = np.random.randn(nc,nh) * np.sqrt(1/nh)

		#Initial Hidden state
		self.state = np.zeros((nh,1))

	def forward_pass(self,X,vocabulary):

		outputs = []	
		hidden_states = []

		#Sequence is a list of integers corresponding to character in the text
		#len(X)+1 because we need a <start> token
		sequence = [0]*(len(X)+1)


		for i,xt in enumerate(X):
			sequence[i+1] = vocabulary.index(xt)

		#print("Sequence = ",sequence)

		for t in sequence:
			
			#Update the hidden state

			self.state = np.tanh(self.Wx[:,t].reshape(self.nh,1) + np.dot(self.Wh,self.state))

			#Calculate Output
			yt = np.dot(self.Wy,self.state)
			
			#print("shape of (state,yt) =({0},{1})".format(self.state.shape,yt.shape))			
			#Squash to provide a probability distribution between 0 and 1
			ot = softmax(yt)

			#print("ot =",ot)
			#prediction = np.argmax(ot,axis=1)
			outputs.append(ot)
			hidden_states.append(self.state)

		return hidden_states,outputs,sequence

	def backprop(self,x,h,p,target):

		#All derivatives of Cost i.e dWx represents--> dC/dWx 
		dWx = np.zeros(self.Wx.shape)
		dWh = np.zeros(self.Wh.shape)
		dWy = np.zeros(self.Wy.shape)

		dh = np.zeros(self.state[0])

		#danext = (1 - self.state[t]**2)*dh

		danext = np.zeros(self.state[0].shape)

		for t in range(len(sequence)):

			dy = np.copy(p)

			dy[target[t]] -= 1

			dh = np.dot(self.Wy.T,dy) + np.dot(Wh.T,danext)

			da = (1 - (self.state[t])**2)*dh

			danext = da

			dWy += np.dot(dy,self.state[t].T)

			dWx += np.dot(da,x[t].T)

			dWh += np.dot(da, self.state[t-1].T)

		return dWx, dWh, dWy

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

		hidden_states,output,sequence = my_rnn.forward_pass(line,vocabulary)

		#append the <end> token in target!
		sequence.append(len(vocabulary)-1)

		loss = calc_cost(output,sequence[1:])
		print("Cost = {}".format(loss))
		
		# print("Predictions Start here")
		# for char in output:
		# 	prediction = predict(char)
		# 	#print("Prediction =",pred)
		# 	#print(vocabulary[prediction],end="")
		# print("")

if __name__ == '__main__':
	main()