"""
This script contains the definition of parameters for the traitment of explosions
recorded on thermal video (Optris format .ravi).
The namedtuple Video contains the path to read the video file, the pixel size (which depends on the optics and
distance camera-object), and a list of explosions (namedtuples) recorded in the video file
The namedtuple Explosion contains the id label for the explosion, the video path, initial and end time in framenumbers,
the vertical position of the region_of_interest (in pixels), and the time (in framenumbers) at which the plume front crosses the upper roi.
"""

from collections import namedtuple

Explosion = namedtuple("Explosion", ["id","video", "ini_time", "end_time", "roi_1", "roi_2", "r2x_time"])
Video = namedtuple("Video", ["path", "pixel_size", "explosions"])
disc = "/Volumes/SHARED/Data/thvid"

videos = (disc+"/trim/20210819_1208.ravi",
          disc+"/trim/20211021_0945.ravi",
          disc+"/trim/20211022_0850.ravi",
          disc+"/trim/ir_21042109_20220602_080806.ravi",
          disc+"/trim/20220602_0952.ravi",
          disc+"/trim/20220707_1224.ravi",
          disc+"/trim/20220707_1242.ravi",
          disc+"/trim/20220816_0838.ravi",
         )

# 2021_08
# Explosion["ID","VIDEO","I_t","E_t","ROI_1","ROI_2","R2t"]
E041 = Explosion("E041", videos[ 0], 248, 3090, 534, 499, 390)
E042 = Explosion("E042", videos[ 1], 256, 8056, 338, 303, 296)
E043 = Explosion("E043", videos[ 2], 190, 9100, 337, 302, 350)
E044 = Explosion("E044", videos[ 3], 1, 3090, 448, 413, 92)
E045 = Explosion("E045", videos[ 4], 164, 3600, 448, 413, 346)
E046 = Explosion("E046", videos[ 5], 10, 1800, 442, 407, 124)
E047 = Explosion("E047", videos[ 6], 242, 1590, 448, 413, 600)
E048 = Explosion("E048", videos[ 7], 340, 1836, 448, 413, 730)




####**** Videos ****
# 2021_08
v00 = Video(videos[ 0], 2.867, [E041])
v01 = Video(videos[ 1], 2.867, [E042])
v02 = Video(videos[ 2], 2.867, [E043])
v03 = Video(videos[ 3], 2.867, [E044])
v04 = Video(videos[ 4], 2.867, [E045])
v05 = Video(videos[ 5], 2.867, [E046])
v06 = Video(videos[ 6], 2.867, [E047])
v07 = Video(videos[ 7], 2.867, [E048])

