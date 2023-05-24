"""
Script created to process explosions recorded on thermal video files (Optris format .ravi).
It obtains the total volume at the crater's level and at a level 100m higher

@author fvasconez
@created 2022.08.22

@modified 2023.04.06
    Automatically process all the videos contained in a tupple
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as poly
import cv2 as cv
import sys
import defExplosions as exp
from collections import namedtuple

# counting from video #33 we restart having all of VIGIA in another definitions file
video = exp.v09
showPlot = False
manualVel = False

# showing params
MAX_LEN = int(32*1)
mean_vals = np.zeros((MAX_LEN,))
m_ii = 0
color = cv.COLORMAP_INFERNO

# func params
exploding = False
roi_1 = video.explosions[0].roi_1    #399
roi_2 = video.explosions[0].roi_2    #350
Void = namedtuple("void", ["l", "r"])
void = Void(80,80)
clicked = False
ini_coords = []
full = False
ini_time = 0
end_time = 0
cross_r2 = False

# other params
H = video.pixel_size   # m/pixel
mx_grad_1 = []
last_h = roi_1
last_cml = np.zeros((roi_1,))

# results

#***************** Functions ****************
def calibration():
    #LUT_file = "/Volumes/SHARED/Califiles/Kennlinie-16080094-15-0-250.prn"
    LUT_file = "/Volumes/SHARED/Califiles/Kennlinie-21042109-15-0-250.prn"
    caldata = np.genfromtxt(LUT_file)
    x = caldata[:,0]
    y = caldata[:,1]
    coef,residuals = poly.fit(x,y,24,full=True)
    return coef

def scale(image, S=7):
    global m_ii
    global mean_vals
    mean_vals[m_ii] = image.mean()
    m_ii += 1
    m_ii %= MAX_LEN
    im_mean = mean_vals.mean()
    im_stdv = image.std()
    im_max = np.min([image.max(), im_mean+S*im_stdv])
    im_min = np.max([image.min(), im_mean-S*im_stdv])
    clip_im = np.clip(image,im_min,im_max)-im_min
    return((255*(clip_im/(im_max-im_min))).astype(np.uint8),[cal(im_min),cal(im_max)])

def clickCoordinate(event, x, y, flags, param):
    global clicked
    global ini_coords
    if(event == cv.EVENT_LBUTTONDOWN):
        clicked = not clicked
        ini_coords = [x, y]
        print("Ini:",x,y)
    if event == cv.EVENT_LBUTTONUP :
        if clicked:
            print("End:",x,y)
            
def clickPlot(event):
    global clicked
    global ini_coords
    if(event.name == "button_press_event"):
        clicked = not clicked
        ini_coords = [event.x, event.y]
        print("Ini:",event.x,event.y)
    if event.name == "button_release_event" :
        if clicked:
            print("End:",event.x,event.y)
        
def get_lims(tsrs_):
    global exploding
    global void
    tsrs = tsrs_ - tsrs_.min()
    md = np.median(tsrs)
    st = tsrs.std()
    delta = tsrs.max()-md
    #clp = np.flatnonzero(np.clip(tsrs-(md+max(st,delta/2)),0,1000))
    clp = np.flatnonzero(np.clip(tsrs-(md+st),0,1000))
    if clp.size < 2 or not exploding: 
        lims = np.array([-void.l,-void.l])
    else:
        lims = np.array([clp[0],clp[-1]])
    return lims

def get_CM(tsrs):
    if tsrs.size < 2: return 0
    tsrs = tsrs - tsrs.min()
    ii = np.arange(tsrs.size)
    sigma = np.sum(tsrs)
    return int(np.dot(tsrs,ii)/sigma)

def get_volume(diameter):
    d = np.array(diameter)
    Vtot = np.pi * H * np.dot(d,d)/4
    return int(Vtot)

def get_height(cm_line):
    global exploding
    global last_cml
    global last_h
    if not exploding : return int(last_h)
    # take out the background
    c_line = cm_line - last_cml
    # obtain the median and std of the "clean" portion
    md = np.median(c_line) if c_line.size > 1 else 0
    st = c_line.std() if c_line.size > 1 else 0
    delta = c_line.max() - md
    sqr = np.clip(c_line-2*md,0,4*md)
    cl_prof = np.flatnonzero(np.clip(c_line - (md+min(st,delta/8)),0,1000))
    last_cml = cm_line

    clp = np.append(cl_prof, 0)
    return clp[0]
    
# Generate the colorbar
def gen_colorbar(frame,lims,row=144,col=12):
    X,Y = [12, 256]
    lw_val = np.around(lims[0],1)
    hg_val = np.around(lims[1],1)
    scale = np.ones((X,Y),np.uint8)
    scale[:,:] = np.linspace(255,0,256)
    c_scale = cv.applyColorMap(scale.T, cv.COLORMAP_INFERNO)
    frame[row:(row+Y),col:col+X] = c_scale
    cv.putText(frame,str(hg_val),(col-2,row-8),cv.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1,cv.LINE_AA)
    cv.putText(frame,str(lw_val),(col-2,row+Y+16),cv.FONT_HERSHEY_SIMPLEX,0.4,(2555,255,255),1,cv.LINE_AA)
    cv.rectangle(frame, (col-1,row-1),(col+X,row+Y),(128,128,128),1)
    return(scale.transpose())    

    
#***************** Main *********************
vd_therm = [
    #exp.v00,
    exp.v01,
    exp.v02,
    exp.v03,
    exp.v04,
    exp.v05,
    exp.v06,
    exp.v07,
    exp.v08,
    exp.v09,]
    
for video in vd_therm:
    diamtr_1 = []
    diamtr_2 = []

    c_line = []
    bgnd_1 = []
    bgnd_2 = []
    prof_1 = []
    prof_2 = []
    height = []
    mx_tmp = []
    hrzntl = []
    ts_f_1 = []
    ts_f_2 = []
    ts_cnt = 0
    ivel = True

    vc = cv.VideoCapture(video.path)
    vc.set(cv.CAP_PROP_FORMAT, -1)

    cols  = int(vc.get(cv.CAP_PROP_FRAME_WIDTH))  # Get video frames width
    rows = int(vc.get(cv.CAP_PROP_FRAME_HEIGHT))  # Get video frames height
    #rows = 480
    #cols = 640
    if cols == 0 or rows == 0:
        print("Video not read")
        exit(1)

    cv.namedWindow("Reventador")
    cv.setMouseCallback("Reventador",clickCoordinate)

    cal = calibration()
    # retrieve the initial times from db
    ini_time = [xp.ini_time for xp in video.explosions]
    end_time = [xp.end_time for xp in video.explosions]
    r2x_time = [xp.r2x_time for xp in video.explosions]
    bgnd_l = np.zeros((rows-1,))
    print(video.path)
    print("Total frames:",vc.get(cv.CAP_PROP_FRAME_COUNT))
    while True:
        ret,r_frame = vc.read()
        if not ret: break
        if ini_time[0] > 480 :
            if MAX_LEN < vc.get(cv.CAP_PROP_POS_FRAMES) <= (ini_time[0] - 256): continue

        # verify if a catalogged explosion is in progress
        if vc.get(cv.CAP_PROP_POS_FRAMES) in ini_time:    
            exploding = True
            i_xp = ini_time.index(vc.get(cv.CAP_PROP_POS_FRAMES))
            roi_1 = video.explosions[i_xp].roi_1
            roi_2 = video.explosions[i_xp].roi_2
            
        # to do when the explosion is finished         
        if vc.get(cv.CAP_PROP_POS_FRAMES) in end_time:
            exploding = False
            vol_1 = get_volume(diamtr_1)
            vol_2 = get_volume(diamtr_2)
            print("Volume",vol_1,"m^3 in explosion",video.explosions[i_xp].id,(video.explosions[i_xp].end_time-video.explosions[i_xp].ini_time)/32,"s")
            print("Vol_2",vol_2,"m^3 in explosion",video.explosions[i_xp].id,(video.explosions[i_xp].end_time-video.explosions[i_xp].r2x_time)/32,"s")
            print("Ent. coef. =",np.round(vol_1/vol_2,4))
            if showPlot:
                plt.plot(diamtr_1)
                plt.plot(diamtr_2)
                plt.show()


            np.savez("diams/"+video.explosions[i_xp].id+"_diameter",diam_1=diamtr_1, diam_2=diamtr_2,t_spots=[video.explosions[i_xp].ini_time, video.explosions[i_xp].end_time, video.explosions[i_xp].r2x_time], maxs=hrzntl)
            diamtr_1 = []
            diamtr_2 = []

        if vc.get(cv.CAP_PROP_POS_FRAMES) in r2x_time: cross_r2 = True
        # put the frame in the correct shape, taking out the first row (not image info)
        frame = r_frame.view(np.int16).reshape(rows,cols)[1:,:]
        r_frame,lims = scale(frame)
        g_frame = cv.rotate(r_frame,cv.ROTATE_90_COUNTERCLOCKWISE)


        #mx_tmp.append(np.max(cal(frame[:,void.l:-void.r]),axis=1))
        hrzntl.append(np.max(cal(frame[:,void.l:-void.r]),axis=0))

        if vc.get(cv.CAP_PROP_POS_FRAMES) <= MAX_LEN:
            bgnd_1.append(g_frame[roi_1,void.l:-void.r])
            bgnd_2.append(g_frame[roi_2,void.l:-void.r])
            bgnd_l += (np.mean(frame,axis=1)/MAX_LEN)
            prof_1.append(g_frame[roi_1,void.l:-void.r])
            continue
        if not full:
            bgnd_1 = np.array(bgnd_1)
            bgnd_2 = np.array(bgnd_2)
            bgnd_1m = np.median(bgnd_1,axis=0)
            dt_bg = (bgnd_1m.max()-bgnd_1m.min())
            full = True

        # obtain the timeseries
        tsr1_r = g_frame[roi_1,void.l:-void.r].copy()-bgnd_1m  
        tsr2_r = g_frame[roi_2,void.l:-void.r].copy()-bgnd_1m
        prof_1.append(g_frame[roi_1,void.l:-void.r])
        prof_1.pop(0)
        tsr1 = cal(tsr1_r)
        tsr2 = cal(tsr2_r)

        lim1 = get_lims(tsr1)+void.l
        lim2 = get_lims(tsr2)+void.l


        ### Getting the contours to limit the plume
        # ========================
        # Try to find the x largest and keep only them. Draw them directly on the picture
        bw = cv.Canny(g_frame[:roi_1,void.l:-void.r],32,60)
        contours, hier = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        len_conts = sorted(contours, key=len, reverse=True)

        if ivel:
            ini_vel = 0
        ts_f_1.append(frame[void.l:-void.r,cols-roi_1].copy().max())
        ts_f_2.append(frame[void.l:-void.r,cols-(roi_1-1)].copy().max())
        ts_cnt += 1
        if exploding:

            if ts_cnt >= 268 and ivel and manualVel:
                d_time = np.argmax(np.gradient(np.array(ts_f_2)))-np.argmax(np.gradient(np.array(ts_f_1)))
                ini_vel = np.around(H*32/d_time,1)
                print("Ini.vel.=",ini_vel)
                plt.figure("Grad")
                plt.plot(np.gradient(np.array(ts_f_1)),label="T1")
                plt.plot(np.gradient(np.array(ts_f_2)),label="T2")
                plt.legend()
                plt.grid()
                fig = plt.figure("TSer")
                cid = fig.canvas.mpl_connect('button_press_event', clickPlot)
                plt.plot(np.array(ts_f_1),label="T1")
                plt.plot(np.array(ts_f_2),label="T2")
                plt.legend()
                plt.grid()
                plt.show()
                ts_f_1=[]
                ts_f_2=[]
                ivel = False
            diamtr_1.append(lim1[1]-lim1[0])
            if not cross_r2: diamtr_2.append(0)


        c_frame = cv.applyColorMap(g_frame,color)
        cv.putText(c_frame,"File elaps.: "+str(np.around(vc.get(cv.CAP_PROP_POS_FRAMES)/32,1))+" s",(int(rows-120),int(cols-20)),cv.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1,cv.LINE_AA)
        if exploding:
            cv.putText(c_frame,"Explosion "+video.explosions[i_xp].id+":",(10,15),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv.LINE_AA)
            cv.putText(c_frame,"Elapsed : "+str(np.around((vc.get(cv.CAP_PROP_POS_FRAMES)-video.explosions[i_xp].ini_time)/32,1))+" s",(10,35),cv.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1,cv.LINE_AA)
            cv.putText(c_frame,"Exit vel. : "+str(ini_vel)+" m/s",(10,50),cv.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1,cv.LINE_AA)


        ## draw the limits of the plume
        # in roi_1
        cv.line(c_frame,(lim1[0],roi_1-20),(lim1[0],roi_1+20),(200,200,200),2)
        cv.line(c_frame,(lim1[1],roi_1-20),(lim1[1],roi_1+20),(200,200,200),2)


        # in roi_2
        if cross_r2:
            diamtr_2.append(lim2[1]-lim2[0])
            cv.line(c_frame,(lim2[0],roi_2-10),(lim2[0],roi_2+10),(200,200,200),2)
            cv.line(c_frame,(lim2[1],roi_2-10),(lim2[1],roi_2+10),(200,200,200),2)
            cv.putText(c_frame,"Crossing ROI_2",(500,15),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv.LINE_AA)



        # show the frame number
        cv.putText(c_frame,str(vc.get(cv.CAP_PROP_POS_FRAMES)),(15,int(cols-20)),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv.LINE_AA)


        # draw the roi
        c_frame[roi_1,void.l:-void.r] = 150
        c_frame[roi_2,void.l:-void.r] = 200
        #gen_colorbar(c_frame,lims)
        # include the contours
        cv.drawContours(c_frame[:roi_1,void.l:-void.r],len_conts[:8],-1,(255,255,255),1)

        cv.putText(c_frame,"ROI_1",(int(void.l),int(roi_1)-3),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv.LINE_AA)
        cv.putText(c_frame,"ROI_2",(int(void.l),int(roi_2)-3),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv.LINE_AA)
        if len(diamtr_1) > 0: 
            cv.putText(c_frame,"D1="+str(np.around(diamtr_1[-1]*H,1))+" m",(int(void.l+tsr1_r.size+10),int(roi_1)-3),cv.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1,cv.LINE_AA)
            cv.putText(c_frame,"D2="+str(np.around(diamtr_2[-1]*H,1))+" m",(int(void.l+tsr2_r.size+10),int(roi_2)-3),cv.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1,cv.LINE_AA)


        cv.imshow("Reventador",c_frame)

        k = cv.waitKey(1)

        # Pause pressing "P"
        if k == ord('p') or k == ord('P'):
            k = cv.waitKey()
        if k == ord('d') or k == ord('D'):
            plt.plot(diamtr_1)
            plt.plot(diamtr_2)
            plt.show()

        if k == ord('v'):
            plt.figure("Grad")
            plt.plot(np.gradient(np.array(ts_f_1)),label="T1")
            plt.plot(np.gradient(np.array(ts_f_2)),label="T2")
            plt.legend()
            plt.grid()
            fig = plt.figure("TSer")
            cid = fig.canvas.mpl_connect('button_press_event', clickPlot)
            plt.plot(np.array(ts_f_1[-64:]),label="T1")
            plt.plot(np.array(ts_f_2[-64:]),label="T2")
            plt.legend()
            plt.grid()
            plt.show()
        if k== ord('x'):
            exploding = not exploding
        # Exit pressing "Q"
        if k == ord('q') or k == ord('Q'):
            break

    #cv.waitKey()
    cv.destroyAllWindows()