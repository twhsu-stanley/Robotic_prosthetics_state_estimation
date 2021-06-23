## This script plots real-time data from a server (typically a 
## RPi) using UDP with any Wi-Fi connection. Socket ports being used 
## are 5012-5024.
##
## The rate at which the plots refresh depends on computer speed and load. 
## The less plots being plotted, the faster the refresh rate. 
##
## Script requirements: Python version 3.7, PyQtGraph, and PyQt5
##
## Comments with ## before them below indicate user can change to 
## his/her preference

print('=============================================')
print('Real-time data plotter using UDP')
print('Created by A. Ma & U. Lee 2020 for the ')
print('Neurobionics Lab, UMichigan')
print('=============================================')
print('This will automatically detect how many')
print('data streams are being sent to pre-defined')
print('sockets and plot accordingly.')
print('')

from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.ptime import time as tim
import pyqtgraph as pg
import numpy as np
import socket
import sys
import struct
import time

input("Press Enter to start...")
print('')

app = QtGui.QApplication([])
app.setWindowIcon(QtGui.QIcon('Michigan_logo.png'))

# Setting background and text color before creating the widget
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOptions(useOpenGL=True)
win = pg.GraphicsWindow(title="Neurobionics Lab Receiver")
win.showMaximized()

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=False)  # set to true for smoother lines (expensive process)

numPoints = 150 ## number of max data points on each plot

class sock:
	# create socket object 
	def __init__(self,port):
		self.port = port
		soc = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
		self.soc = soc
		server_address = (socket.gethostbyname(socket.gethostname()), self.port)
		soc.bind(server_address)
		soc.setblocking(0)
		#print(sys.stderr, 'starting up on %s port %s' % server_address)
	def read_data(self):
		data , address =self.soc.recvfrom(4096)
		data_tuple = struct.unpack('d',data)
		temp_data = data_tuple[0]
		return temp_data
	def read_str(self):
		#data, address =self.soc.recvfrom(4096)
		temp_data = self.soc.recvfrom(4096)[0].decode()
		return temp_data
	def block(self):
		self.soc.setblocking(1)	

tickFont = QtGui.QFont()
tickFont.setPointSize(25)

# create sockets for data streams
soc0 = sock(5012)
soc1 = sock(5013)
soc2 = sock(5014)
soc3 = sock(5015)
soc4 = sock(5016)

# create sockets for y axis labels
soc5 = sock(5017)
soc6 = sock(5018)
soc7 = sock(5019)
soc8 = sock(5020)

# create sockets for y axis labels' units
soc9  = sock(5021)
soc10 = sock(5022)
soc11 = sock(5023)
soc12 = sock(5024)

time.sleep(0.5) # gives extra time for sockets to be created

# Making the first plot
p1 = win.addPlot() ## setting plot name
#p1.setTitle('',**{'size': '40pt'})
curve1 = p1.plot(pen=pg.mkPen(width = 10, color=(217, 83, 25))) ## setting plot color
ylabel1 = soc5.read_str()
ylabelU1 = soc9.read_str()
p1.setLabel('left', ylabel1, units = ylabelU1, **{'font-size':'30pt'}) ## setting y-axis label and units
p1.getAxis('left').tickFont = tickFont ## setting tick font
p1.getAxis('bottom').setStyle(showValues=False)
p1.getAxis('left').setStyle(textFillLimits=[(0,0.2)])
datay1 = [0]*numPoints ## initializing with all zeros on plot	
p1.enableAutoRange('y', True) # auto rescale of y-axis
p1.getAxis('left').setWidth(140)
p1.showGrid(y=True, alpha = 0.5)

lastTime = tim() # used for frames per second (FPS) count
fps = None # frames per second (FPS) variable
ptr = 0 # main loop counter
numGraphs = 1 # counter used to keep track of how many graphs will be plotted

try:
	soc2.read_data()
	p2 = win.addPlot() # creating the graph
	#p2.setTitle('',**{'size': '40pt'})
	curve2 = p2.plot(pen=pg.mkPen(width = 10, color=(0, 114, 189)))
	ylabel2 = soc6.read_str()
	ylabelU2 = soc10.read_str()
	p2.setLabel('left', ylabel2, units = ylabelU2, **{'font-size':'30pt'}) 
	p2.getAxis('left').tickFont = tickFont 
	p2.getAxis('bottom').setStyle(showValues=False)
	datay2 = [0]*numPoints
	p2.enableAutoRange('y', True)
	p2.getAxis('left').setWidth(140) 
	p2.getAxis('left').setStyle(textFillLimits=[(0,0.2)])
	p2.showGrid(y=True, alpha = 0.5)
	numGraphs+=1
except socket.error:
	print('no plot 2')

# checking to see if data for graph 3 is being transmitted	
try:
	soc3.read_data()
	win.nextRow() # creates new row of plots
	p3 = win.addPlot() # creating the graph
	#p3.setTitle('',**{'size': '40pt'})
	curve3 = p3.plot(pen=pg.mkPen(width = 10, color=(162, 20, 47)))
	ylabel3 = soc7.read_str()
	ylabelU3 = soc11.read_str()
	p3.setLabel('left', ylabel3, units = ylabelU3, **{'font-size':'30pt'})
	p3.getAxis('left').tickFont = tickFont 
	p3.getAxis('bottom').setStyle(showValues=False)
	datay3 = [0]*numPoints
	p3.enableAutoRange('y', True)
	p3.getAxis('left').setWidth(140)
	p3.getAxis('left').setStyle(textFillLimits=[(0,0.2)])
	p3.showGrid(y=True, alpha = 0.5)
	numGraphs+=1
except socket.error:
	print('no plot 3')
	
# checking to see if data for graph 4 is being transmitted
try:
	soc4.read_data()
	p4 = win.addPlot() # creating the graph
	#p4.setTitle('',**{'size': '40pt'})
	curve4 = p4.plot(pen=pg.mkPen(width = 10, color=(126, 47, 142), style=QtCore.Qt.SolidLine))
	ylabel4 = soc8.read_str()
	ylabelU4 = soc12.read_str()
	p4.setLabel('left', ylabel4, units = ylabelU4, **{'font-size':'30pt'}) 
	p4.getAxis('left').tickFont = tickFont 
	p4.getAxis('bottom').setStyle(showValues=False)
	datay4 = [0]*numPoints
	p4.enableAutoRange('y', True)
	p4.getAxis("left").setWidth(140)
	p4.getAxis('left').setStyle(textFillLimits=[(0,0.2)])
	p4.showGrid(y=True, alpha = 0.5)
	numGraphs+=1
except socket.error:
	print('no plot 4')
	

# setting soc0-soc4 to blocking.
# we don't care to set the rest to blocking
# because they won't be used anymore
soc0.block()
soc1.block()
soc2.block()
soc3.block()
soc4.block()

counter = 15	# counter that determines how many data points to receive 
				# before updating the plot

# these will be used to check if there is any delay between the server 
# sending the data and plotting
start_time_receiver = time.time()
start_time_sender = soc0.read_data()

def update():
	global ptr, counter, fps, lastTime, curve1, curve2, curve3, curve4, datay1, datay2, datay3, datay4, p1, p2, p3, p4
	
	elapsed_time_receiver = time.time()-start_time_receiver
	sender_time = soc0.read_data()
	elapsed_time_sender = sender_time-start_time_sender
	lag = elapsed_time_receiver-elapsed_time_sender
	
	ys1 = soc1.read_data() # reading in plot 1 data
	# reading in other plots' data if the plot was made
	if numGraphs > 1:
		ys2 = soc2.read_data()
	if numGraphs > 2:
		ys3 = soc3.read_data()
	if numGraphs > 3:
		ys4 = soc4.read_data()
	
	datay1.append(ys1)
	if numGraphs > 1:
		datay2.append(ys2)
	if numGraphs > 2:
		datay3.append(ys3)
	if numGraphs > 3:
		datay4.append(ys4)
	
	datay1 = datay1[-numPoints:]
	if numGraphs > 1:
		datay2 = datay2[-numPoints:]
	if numGraphs > 2:
		datay3 = datay3[-numPoints:]
	if numGraphs > 3:
		datay4 = datay4[-numPoints:]
	
	# evaluating if counter needs to be changed based on how much latency 
	# there is between server sending and client plotting. the larger the 
	# latency, the less often the plots are refreshed in order to alleviate
	# receiver computer load, allowing the plots to catch up to real-time
	if ptr%300==0:
		if lag > 0.1:
			counter+=2
		elif counter>1:
			counter-=1
	
	if ptr%counter==0:
		# updating plots
		curve1.setData(datay1)
		if numGraphs > 1:
			curve2.setData(datay2)
		if numGraphs > 2:
			curve3.setData(datay3)
		if numGraphs > 3:
			curve4.setData(datay4)
	
		# FPS/refresh rate calulator
		now = tim()
		dt = now - lastTime
		lastTime = now
		if fps is None:
			fps = 1.0/dt
		else:
			s = np.clip(dt*3., 0, 1)
			fps = fps * (1-s) + (1.0/dt) * s
		print('FPS:', fps)
		
	ptr+=1
		
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)	# amount of pause between each run of update, set to 0 for 
				# fastest possible refresh

# Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()