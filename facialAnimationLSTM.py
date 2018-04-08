import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.utils.data as data_utils

#importing data from mat file
import scipy.io
from sys import getsizeof
matData = scipy.io.loadmat('starter_data.mat') #reading input from mat file into dictionary

#storing train and test data separately
X_train = matData['X_train']
all_x_test = matData['X_test']
Y_train = matData['Y_train']
all_y_test = matData['Y_test'] 

def prepare_sequence(seq):
	tensor = torch.FloatTensor(seq)
	return autograd.Variable(tensor)

#declaring LSTM model
class LSTMTagger(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        # self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim) #linear layer
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 25, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 25, self.hidden_dim)))

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        out = self.fc(lstm_out[:, -1, :])
        return out

if __name__ == '__main__':

	#printing list of keys in dictionary file
	for key in matData:
		print(key)

	#forming test data
	x_initial = np.concatenate((all_x_test[0][0], all_x_test[0][1]), axis=0) 
	x_test_final = np.concatenate((x_initial, all_x_test[0][2]), axis=0)

	y_initial = np.concatenate((all_y_test[0][0], all_y_test[0][1]), axis=0) 
	y_test_final = np.concatenate((y_initial, all_y_test[0][2]), axis=0)

	#getting dimensions
	num_train_samples, input_dim = X_train.shape
	num_train_samples, output_dim = Y_train.shape

	#initializing model
	hidden_dim = 256
	model = LSTMTagger(input_dim, hidden_dim, output_dim)
	loss_func = nn.MSELoss()
	optimizer = optim.SGD(model.parameters(), lr=0.01)

	#preparing data
	xtrain = np.zeros((49975, 25, 65))
	xtest = np.zeros((2975, 25, 65))
	for i in range(0, 49975):
		xtrain[i, :, :] = X_train[i:i+25, :]
	for i in range(0, 2975):
		xtest[i, :, :] = x_test_final[i:i+25, :]

	ytrain = Y_train[25:len(Y_train), :]
	ytest = y_test_final[25:len(y_test_final), :]

	#preparing test sequence
	x_test_prepared = prepare_sequence(xtest)
	x_test_prepared = x_test_prepared.contiguous()
	y_test_prepared = prepare_sequence(ytest)
	y_test_prepared = y_test_prepared.contiguous()

	print(x_test_prepared.shape)
	print(y_test_prepared.shape)

	#preparing input sequence
	# inputTrain = prepare_sequence(xtrain)
	# outputTrain = torch.FloatTensor(ytrain)
	#trying for all inputs at once
	# outputs = model(inputTrain.view(len(inputTrain), 25, -1))
	# loss = loss_func(outputs, outputTrain)
	# print(loss)

	X = torch.FloatTensor(xtrain)
	Y = torch.FloatTensor(ytrain)

	train_data = data_utils.TensorDataset(X, Y)
	dataloader = data_utils.DataLoader(train_data, batch_size=300, shuffle=True)


	for epoch in range(15):

		model.hidden = model.init_hidden() #detaching lstm from history of last sequence

		for i_batch, sample_batched in enumerate(dataloader):
			print(i_batch)
			model.zero_grad() 
			xtrain = autograd.Variable(sample_batched[0], requires_grad = False)
			ytrain = autograd.Variable(sample_batched[1], requires_grad = False)
			# x = torch.from_numpy(xtrain.numpy()).float()
			# y = torch.from_numpy(ytrain.numpy()).float()
			outputTrain = model(xtrain.view(len(xtrain), 25, -1))
			loss = loss_func(outputTrain, ytrain)
			loss.backward(retain_graph=True)
			optimizer.step()
			
		print('finished %s!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', epoch)

	# testing with test sequence
	outputs = model(x_test_prepared.view(len(x_test_prepared), 25, -1))
	print(outputs.shape)
	loss = loss_func(outputs, y_test_prepared)	
	print(loss)

