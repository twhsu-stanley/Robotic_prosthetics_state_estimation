import sender
import time
import numpy as np

start_time = time.time() 	# for determining lag between sender sending data
							# and receiver plotting data

ptr=0

while 1:
	try:
		elapsed_time = time.time()-start_time
		
		transmit1 = np.sin(elapsed_time*2*3.14)
		transmit2 = np.sin(elapsed_time*2*3.14+1.57)+1
		transmit3 = 0.3*(np.random.random()+0.3)
		transmit4 = 17*np.random.random()
		
		sender.graph(elapsed_time, transmit1, 'Torque', 'Nm', transmit2, 'Velocity', 'm/s', transmit3, 'Mass', 'kg', transmit4, 'Length', 'm')
		
		print('Elapsed time:', elapsed_time, ptr)
		ptr+=1
		
	except KeyboardInterrupt:
		break
