# Model Generator
# generate the regressor model based on a Kronecker Product of different function 
# each function is s one dimensional function of task variables

import numpy as np
import math
import pickle
import scipy

class Basis:
	def __init__(self, n, var_name):
		self.n = n
		self.var_name = var_name

	#Need to implement with other subclasses
	def evaluate(self,x):
		pass

	#Need to implement the derivative of this also
	def evaluate_derivative(self,x):
		pass

	def evaluate_conditional(self, x, apply_derivative):
		if(apply_derivative == True):
			return self.evaluate_derivative(x)
		else:
			return self.evaluate(x)

class Polynomial_Basis(Basis):
	def __init__(self, n, var_name):
		Basis.__init__(self, n, var_name)
		self.size = n

	#This function will evaluate the model at the given x value
	def evaluate(self, x):
		result = [math.pow(x,i) for i in range(0, self.n)]
		return np.array(result)

	#This function will evaluate the derivative of the model at the given x value
	def evaluate_derivative(self, x):
		if self.n == 1:
			result = [0]
		elif self.n > 1:
			result = [0]
			result += [i * math.pow(x,(i-1)) for i in range(1, self.n)]
		return np.array(result)

class Fourier_Basis(Basis):
	def __init__(self, n, var_name):
		Basis.__init__(self, n, var_name)
		self.size = 2*n-1

	#This function will evaluate the model at the given x value
	def evaluate(self, x):
		result = [1]
		result += [np.cos(2*np.pi*i*x) for i in range(1, self.n)]
		result += [np.sin(2*np.pi*i*x) for i in range(1, self.n)]
		return np.array(result)

	#This function will evaluate the derivative of the model at the given x value
	def evaluate_derivative(self, x):
		result = [0]
		result += [-2*np.pi*i*np.sin(2*np.pi*i*x) for i in range(1, self.n)]
		result += [2*np.pi*i*np.cos(2*np.pi*i*x) for i in range(1, self.n)]
		return np.array(result)

class Berstein_Basis(Basis):
	def __init__(self, n, var_name):
		Basis.__init__(self, n, var_name)
		self.size = n + 1

	#This function will evaluate the model at the given x value
	def evaluate(self, x):
		result = [math.comb(self.n, i) * x**i * (1-x)**(self.n-i) for i in range(0, self.n + 1)]
		return np.array(result)

	#This function will evaluate the derivative of the model at the given x value
	def evaluate_derivative(self, x):
		if self.n >= 2:
			result = [-self.n * (1-x)**(self.n-1)]
			result += [math.comb(self.n, i) * (i * x**(i-1) * (1-x)**(self.n-i) - x**i * (self.n-i) * (1-x)**(self.n-i-1)) for i in range(1, self.n)]
			result += [self.n * x**(self.n-1)]
		elif self.n == 1:
			result = [-1, 1]
		elif self.n == 0:
			result = [0]
		return np.array(result)

#Model Object:
# list of basis objects
# string description
# model_size

class Kronecker_Model:
	def __init__(self, *funcs):
		self.funcs = funcs

		#Calculate the size of the parameter array
		#Additionally, pre-allocate arrays for kronecker products intermediaries 
		# to speed up results
		self.alocation_buff = []
		size = 1
		for func in funcs:
			#Since we multiply left to right, the total size will be on the left 
			#and the size for the new row will be on the right
			print((str(size), str(func.size)))
			self.alocation_buff.append(np.zeros((size, func.size)))
			size = size * func.size

		self.size = size
		self.num_states = len(funcs)
        
	#Evaluate the models at the function inputs that are received
	#The function inputs are expected in the same order as they where defined
	#Alternatively, you can also input a dictionary with the var_name as the key and the 
	# value you want to evaluate the function as the value
	def evaluate(self, *function_inputs, partial_derivative=None):
		
		#Crop so that you are only using the number of states and not the gait fingerprint
		states = function_inputs[:self.num_states]

		#Verify that you have the correct input 
		if(len(states) != len(self.funcs)):
			err_string = 'Wrong amount of inputs. Received:'  + str(len(states)) + ', expected:' + str(len(self.funcs))
			raise ValueError(err_string)

		#if(isinstance(states,dict) == False and isinstance(states,list) == False): 
		#	raise TypeError("Only Lists and Dicts are supported, you used:" + str(type(states)))

		#There are two behaviours: one for list and one for dictionary
		#List expects the same order that you received it in
		#Dictionary has key values for the function var names

		result = np.array([1])
		#Assume that you get a list which means that everything is in order
		for values in zip(states, self.funcs, self.alocation_buff):
			curr_val, curr_func, curr_buf = values
			
			#If you get a dictionary, then get the correct input for the function
			if(isinstance(states, dict) == True):
				#Get the value from the var_name in the dictionary
				curr_val = states[curr_func.var_name]

			#Verify if we want to take the partial derivative of this function
			if(partial_derivative is not None and curr_func.var_name in partial_derivative):
				apply_derivative = True
			else: 
				apply_derivative = False

			#Since there isnt an implementation for doing kron in one shot, do it one by one
			result = fast_kronecker(result, curr_func.evaluate_conditional(curr_val, apply_derivative), curr_buf)

		return result

#Evaluate model 
def model_prediction(model, psi, *input_list, partial_derivative = None):
	result = [model.evaluate(*function_inputs, partial_derivative = partial_derivative) @ psi for function_inputs in zip(*input_list)]
	return np.array(result)

##LOOK HERE 
##There is a big mess with how the measurement model is storing the gait fingerprint coefficients
##They should really just be part of the state vector, the AXIS should be stored internally since that is 
## fixed
class Measurement_Model():
	def __init__(self, *models):
		self.models = models

	def evaluate_h_func(self, Psi, *states):
		h = np.zeros((np.shape(Psi)[0], 1))
		k = 0
		for model in self.models:
			h[k] = model.evaluate(*states) @ Psi[k,:].T
			k = k + 1
		return h

	def evaluate_dh_func(self, Psi, *states):
		H = np.zeros((np.shape(Psi)[0], 4))
		k = 0
		for model in self.models:
			j = 0
			for func in model.funcs:
				H[k, j] = model.evaluate(*states, partial_derivative = func.var_name) @ Psi[k, :].T
				j = j + 1
			k = k + 1
		return H

#Calculate the least squares based on the data
def least_squares(model, output, *data):
	#Get data size
	rows = data[0].shape[0]
	columns = model.size

	# Regressor Matrix
	R = np.zeros((rows, columns))
	counter = 0
	for row in zip(*data):
		R[counter, :] = model.evaluate(*row)
		counter = counter + 1

	# Only make it a np array if it isnt
	if isinstance(output,(np.ndarray)):
		output = np.array(output)

	# linear least square solution
	psi = np.linalg.solve(R.T @ R, R.T @ output)

	return psi

#Save the model so that you can use them later
def model_saver(model, filename):
	with open(filename, 'wb') as file:
		pickle.dump(model, file)

#Load the model from a file
def model_loader(filename):
	with open(filename, 'rb') as file:
		return pickle.load(file)

# Speed up implementation of the kronecker product 
# use outer products if a buffer is provided. This saves the time it takes
# to allocate every intermediate result
def fast_kronecker(a, b, buff=None):
	#If you pass the buffer is the fast implementation
	#139 secs with 1 parameter fit
	if(buff is not None):
		return np.outer(a, b, buff).ravel().copy()

	#Else use the default implementation
	#276.738 secs with 1 param
	else:
		return np.kron(a, b)