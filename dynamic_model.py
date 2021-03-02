import numpy as np
import matplotlib.pyplot as plt

class Dynamic_Model:

	#Initial States is a dictionary that will contain a variable name
	# for every state and the initial value for that state
	def __init__(self):
		pass

	def get_predicted_state(self,current_state,time_step,control_input_u):
		pass



class Gait_Dynamic_Model(Dynamic_Model):

	def __init__(self):
		pass

	#The gait dynamics are an integrator for phase that take into consideration 
	# the phase_dot and the timestep. It is assumed that the phase_dot is the 
	# second element in the state vector
	def f_jacobean(self, current_states, time_step):
		amount_of_states = current_states.shape[0]


		#All the states will stay the same except for the first one
		jacobean = np.eye(amount_of_states)


		#Add the integrator
		jacobean[0][1] = time_step

		return jacobean

	#This is a linar function based on the jacobean its pretty straighforward to calculate
	def f_function(self, current_states, time_step):
		#Essentially we want to calculate dot x = Ax


		jacobean = self.f_jacobean(current_states,time_step)

		return jacobean @ current_states
