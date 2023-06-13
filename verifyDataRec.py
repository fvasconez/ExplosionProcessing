# Script created to verify the recorded thermal videos
# @author fvasconez
# @created 2022.01.18
# @modified 2023.01.16  Allow to record a video in .mp4 format
#
import numpy as np
import ctypes as ct
import cv2
import sys
import ind_checkPictures as col
import time
import os

class Evo(ct.Structure):
	_fields_ = [("a", ct.c_uint),
    			("b", ct.c_uint),
                ("c", ct.c_longlong),
                ("d", ct.c_longlong),
                ("e", ct.c_int),
                ("f", ct.c_float),
                ("g", ct.c_float),
                ("h", ct.c_float),
                ]

""" This is the structure of METADATA for each frame
typedef struct __IRDIRECTSDK_API__ EvoIRFrameMetadata
{
  unsigned int counter;     /*!< Consecutively numbered for each received frame */
  unsigned int counterHW;   /*!< Hardware counter received from device, multiply with value returned by IRImager::getAvgTimePerFrame() to get a hardware timestamp */
  long long timestamp;      /*!< Time stamp in UNITS (10000000 per second) */
  long long timestampMedia;
  EvoIRFlagState flagState; /*!< State of shutter flag at capturing time */
  float tempChip;           /*!< Chip temperature */
  float tempFlag;           /*!< Shutter flag temperature */
  float tempBox;            /*!< Temperature inside camera housing */
} EvoIRFrameMetadata;
"""     
    
## First part: Metadata
## ====================
#
# Check if there are gaps in the video verifying the counter for each frame
# Report frame numbers when there are gaps
def verifMetadata(fname, makeAVI=False):
	print("makeAvi",makeAVI)
	data = np.memmap(fname,dtype=Evo(),mode='r')

	ll = data[0]
	print("Verifying metadata. File:",fname)
	print("T.size:",data.size)
	print("Frames:",data.shape[0])
	print("First:",ll)
	print("Last:",(data[-1]))
	cc = 0
	for ee in data[1:] :
		if not (ee[0]-ll[0]) == 1 :
			print(cc,":",ll[0],"-",ee[0])
		cc += 1
		ll = ee
 
  
## Second part: Thermal frames
## ===========================
#
# Join the chunks of the thermal video if not done before and show the video
# If indicated, make a .mp4 video as output
def verifThermal(args, makeAVI=False, fast=True):  
    skip = False
    avi_reduct = 8
    fname = args
    prefix, suffix = os.path.splitext(fname)
    
    data = np.memmap(fname,dtype=np.uint16,mode='r').reshape(480*640,-1)
    mdat = np.memmap(fname.replace('THVID','METAD'),dtype=Evo(),mode='r')
    newShape = data.shape
    
    if len(prefix.split("_")[-1]) == 1:
        # create a new array to put the whole video inside
        newFile = np.memmap(prefix[:-1]+suffix,mode='w+',dtype=np.uint16,shape=(newShape[0],newShape[1]*4))
        newMtdt = np.memmap(prefix[:-1].replace('THVID','METAD')+suffix,mode='w+',dtype=Evo(),shape=(newShape[1]*4,))
    
        print("NewFile.shape:",newFile.shape)
        newFile[:,:newShape[1]] = data
        newMtdt[:newShape[1]] = mdat
        # read the other 3 chunks of the video
        for ii in [2, 3, 4]:
            newFile[:,(ii-1)*newShape[1]:(ii)*newShape[1]] = np.memmap(prefix[:-1]+str(ii)+suffix,mode='r',dtype=np.uint16).reshape(480*640,-1)
            newMtdt[(ii-1)*newShape[1]:(ii)*newShape[1]] = np.memmap(prefix[:-1].replace('THVID','METAD')+str(ii)+suffix,mode='r',dtype=Evo())

    else: # this means that the chunks have already been joined in one single file
        newFile = data
        newMtdt = mdat
    
    print("Verifying thermal frames. File:",fname)
    print("T.size:",data.shape)
	
    print("Frames:",data.size/307200)

    cc = 0
    f_date = time.strftime("VIGIA_%Y%m%d_%H%M%S",time.gmtime((mdat[0])[2]/col.tstamp_factor))
    print(f_date)

    if makeAVI:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f_date+'.mp4', fourcc, 32.0*2/avi_reduct, (480,640), True)
    
    for ii in range(newFile.shape[1]):
        if not skip:
            r_frm = newFile[:,ii].reshape(480,640)
            backgnd = ((r_frm[5:15,70:80].mean()+r_frm[465:475,70:80].mean())/2).astype(np.uint16)
            frame = col.makeColored(r_frm,newMtdt[ii], False)
            cv2.imshow("Recorded",frame)

            if makeAVI and ii % avi_reduct == 0 : out.write(frame)

            ret = cv2.waitKey(1)
            
            if ret == ord('q'):
                break
        skip = not skip
        if not fast: skip = False
                
    if makeAVI: out.release()

    
## =========================
## ========= MAIN ==========
## =========================
if __name__=='__main__':
    if len(sys.argv) == 1:
        print("Too few arguments: At least a file name should be provided.")
        exit(1)
    elif len(sys.argv) == 2:
        make_avi = False
        fl_name = sys.argv[1]
    if len(sys.argv) > 2:
        fl_name, make_avi = sys.argv[1:]
    if "METAD" in fl_name:
        verifMetadata(fl_name, make_avi)
    elif "THVID" in fl_name:
        verifThermal(fl_name, make_avi)
    else:
        print("File not recognized")
        exit(2)