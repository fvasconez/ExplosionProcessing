################################################ ##
### Real time data plotter in opencv (python)   ###
## Plot multiple data for debugging and analysis ##
## Contributors - Vinay @ www.connect.vin        ##
## For more details, check www.github.com/2vin   ##
##                                               ##
## This script was modified to work with thermal ##
## data (obtained with Optris camera).           ##
## @modified 2022.09.07                          ##
## @author fvasconez                             ##
###################################################

import cv2
import numpy as np
from numpy.polynomial import Polynomial as poly
import random

LUT_file = '../Vigia/video/Kennlinie-21042109-15-0-250.prn'
#LUT_file = "/Volumes/SHARED/Califiles/Kennlinie-16080094-15-0-250.prn"
X_compression = 2

# Plot values in opencv program
class Plotter:
    def __init__(self, plot_width, plot_height, num_plot_values):
        self.width = plot_width
        self.height = plot_height
        self.color_list = [(255, 0 ,0), (0, 250 ,0),(0, 0 ,250),
	    			(0, 250, 0),(250, 0 ,250),(250, 250 ,0),
	    			(200, 100 ,200),(100, 200 ,200),(200, 200 ,100)]
        self.color  = []
        self.min_y = 0
        self.max_y = plot_height
        self.val = []
        self.state = []
        self.recalc_count = 0
        caldata = np.genfromtxt(LUT_file)
        self.calib = poly.fit(caldata[:,0], caldata[:,1], 24)
        self.calib = poly.fit([0,1],[0,1],1)
        self.plot_canvas = np.ones((self.height, self.width, 3))*255

        for i in range(num_plot_values*2):
            self.color.append(self.color_list[i])

        # Update new values in plot
    def multiplot(self, val, names, state=False, label = "plot"):
        if int(min(val)) < self.min_y:
            self.recalc_count = 0
            self.min_y = int(min(val))
            
        if int(max(val)) > self.max_y:
            self.recalc_count = 0
            self.max_y = int(max(val))  
            
        self.val.append(val)
        self.state.append(state)
        self.recalc_count += 1
        
        if self.recalc_count > self.width:
            self.max_y = int(np.max([self.height, np.max(self.val)]))
            self.min_y = int(np.min([0, np.min(self.val)]))
        
        while len(self.val) > self.width * X_compression:
            self.val.pop(0)
            self.state.pop(0)

        self.show_plot(label, names)

    # Show plot using opencv imshow
    def show_plot(self, label, names=None):
        self.plot_canvas = np.ones((self.height, self.width, 3))*255
        # Calculate the extreme values for constructing the plot
        a = self.height * self.max_y
        b = self.max_y - self.min_y
        #lb_min = str(np.around(self.lut[self.lut[:,0]==self.min_y][0][1],1))
        #lb_max = str(np.around(self.lut[self.lut[:,0]==self.max_y][0][1],1))
        b_min = np.around(self.calib(self.min_y),1)
        lb_min = str(b_min)
        b_max = np.around(self.calib(self.max_y),1)
        lb_max = str(b_max)
        # counting backwards time
        t_0 = int(-len(self.val)/32)
        # write down the extreme values
        cv2.putText(self.plot_canvas, lb_min, (int(self.width-38),int(self.height-10)),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(self.plot_canvas, lb_max, (int(self.width-38),10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(self.plot_canvas, str(t_0)+" s", (10, int(self.height-10)),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),1,cv2.LINE_AA)
        
        # write the names of the series
        for nn in range(len(names)):
            cv2.putText(self.plot_canvas, names[nn], (10,12+11*nn),cv2.FONT_HERSHEY_SIMPLEX,0.4,self.color[nn],1,cv2.LINE_AA)
        
        # horizontal lines
        E = 1000
        h_lines = np.linspace((int(b_min/E)+(int(b_min*b_max>0)))*E,int(b_max/E)*E,int(b_max/E)-int(b_min/E)+int(b_min*b_max<=0))
        #print(b_min,b_max)
        for hl in h_lines:
            cv2.line(self.plot_canvas, (0,int((a/b)-(self.height*hl/b))), (self.width,int((a/b)-(self.height*hl/b))), (0,0,0), 1)
            #cv2.line(self.plot_canvas, (0,int(hl)), (self.width,int(hl)), (0,0,0), 1)
        
        
        # draw the lines connecting points (i.e., plot the timeseries)
        for i in range(len(self.val)-1):
            for j in range(len(self.val[0])):
                l_color = self.color[j+2] if self.state[i] else self.color[j]
                cv2.line(self.plot_canvas, (int(i/X_compression), int((a/b)-(self.height*self.val[i][j]/b))), (int(i/X_compression)+1, int((a/b)-(self.height*self.val[i+1][j]/b))), l_color, 1)
        
        # show the plot
        cv2.imshow(label, self.plot_canvas)

