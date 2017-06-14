#interesing_view.py
import cv2
import time
import pygame
import imutils
import argparse
import datetime
import random
import numpy as np
from interesion_model import exciting
from pygame.locals import *
from sys import exit

# 创建参数解析器并解析参数
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", help="path to the image file")
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-n", "--name", type = str, default="Capture",help="window name")
ap.add_argument("-w", "--width", type = int,default=800,help="window width")
ap.add_argument("-ht", "--height", type = int,default=1000,help="window height")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")

args = vars(ap.parse_args())
# 如果video参数为None，那么我们从摄像头读取数据
if args.get("video", None) is None and args["image"] is None:
    camera = cv2.VideoCapture(0)
    #等待0.25秒
    time.sleep(0.25)
 
# 否则我们读取一个视频文件
else:
    camera = cv2.VideoCapture(args["video"])

width = args["width"]
height = args["height"]

#faceslen=0
#firstFrame = None
#backgrouds = []
#pedestrians = {} #行人字典

et=exciting(camera,width=width,height=height)

faces = [['陈思羽','18']]
faceShow = []

f=0
while camera.isOpened():

	et.start

	for event in pygame.event.get():
		if event.type == pygame.MOUSEBUTTONDOWN:
			pass
		if event.type == pygame.QUIT:
			#camera.release()
			pygame.quit()
			#exit()

	for i in range(0,len(faceShow)):

		roj = cv2.cvtColor(faceShow[i], cv2.COLOR_RGB2BGR)
		roj = np.swapaxes(roj, 0, 1)
		roj = pygame.pixelcopy.make_surface(roj)
		et.show_text(et.screen,(width-100,10),str(f),(251,116,135),30)
		et.show_text(et.screen,(width,i*200),str(i),(251,65,90),40)

		et.screen.blit(roj, (width, i*200))

	#识别开始
	#差度
	#backgrouds=et.readBackgroud(random.randint(0,19))
	#contours=et.frame_difference(backgrouds,gray)
	#KNN
	KNN=et.KNN_difference(et._frame,args["min_area"])
	if  KNN != []:
		face=et.face(et.gray) #脸
		if face != ():
			for fx,fy,fw,fh in face:
				pygame.draw.rect(et.screen,[255,149,0],[fx,fy,fw,fh],3)
				roi = et.gray[fy:fy+fh,fx:fx+fw]
				roj = et.color[fy:fy+fh,fx:fx+fw]
				roi = cv2.resize(roi,(200,200))
				roj = cv2.resize(roj,(200,200))

				faceShow.append(roj)
				'''if facearray != []:

					x.append(np.asarray(roi,dtype=np.uint8))
					y.append(faceslen)
					et.face_rec([x,y])
					cv2.imwrite('face/face_gray/1/%s.png' % str(faceID),roi)
					cv2.imwrite('face/face_color/1/%s.png' % str(faceID),roj)

					faceslen = faceslen+1
				else:
					et.face2(roj)
'''
				if f<20:
					cv2.imwrite('face/face_gray/1/%s.png' % str(f),roi)
					cv2.imwrite('face/face_color/1/%s.png' % str(f),roj)

					
					f =f+1					
					
				#screen.blit(roi, (0, 0))


				#faceName=et.face2(roi)
				#print(faceName)
				for i in range(0,len(faces)):
					for j in range(0,len(faces[i])):
						pygame.draw.rect(et.screen,[193,133,47],[fx+fw,fy+(42*j),90,40])
						et.show_text(et.screen,(fx+fw,fy+(45*j)),faces[i][j],(255,255,255), True,30)
		
	
		