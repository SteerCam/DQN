import math
from math import sin
from math import cos
from math import tan
from math import sqrt 
from math import atan2



import numpy as np

import sys

np.set_printoptions(threshold=sys.maxsize)


#cam view 4
#width=       768
#height=      576
#cam8_ncx=    7.9500000000e+02
#cam8_nfx=    7.5200000000e+02
#cam8_dx=     4.8500000000e-03
#cam8_dy=     4.6500000000e-03
#cam8_dpx=    5.1273271277e-03
#cam8_dpy=    4.6500000000e-03
#cam8_focal=  1.8579748488e+01
#cam8_kappa1= -1.0419523154e-04
#cam8_cx=     3.5192780774e+02
#cam8_cy=     2.8796348569e+02
#cam8_sx=     1.0804745826e+00
#cam8_tx=     -2.9524368798e+03
#cam8_ty=     1.4102854260e+03
#cam8_tz=     8.0157550249e+04
#cam8_rx=     -1.6296461530e+00
#cam8_ry=     3.6853070326e-01
#cam8_rz=     3.1252104484e+00

#cam view8



#cam8_ncx=    1.6000000000e+03
#cam8_nfx=    1.6000000000e+03
#cam8_dx=     4.6500000000e-03
#cam8_dy=     4.6500000000e-03
#cam8_dpx=    4.6500000000e-03
#cam8_dpy=    4.6500000000e-03
#cam8_focal=  3.4345527587e+00
#cam8_kappa1= 3.2678605363e-02
#cam8_cx=     5.9220854519e+02
#cam8_cy=     3.9728183795e+02
#cam8_sx=     1.0000000000e+00
#cam8_tx=     3.2069863258e+03
#cam8_ty=     2.3119343215e+03
#cam8_tz=     -2.8740583612e+02 
#cam8_rx=     -2.5947507242e+00
#cam8_ry=     1.1385785426e+00 
#cam8_rz=     2.2484536187e+00

#cam view7
cam8_ncx   =                           1.6000000000e+03
cam8_nfx   =                           1.6000000000e+03
cam8_dx    =                           4.6500000000e-03
cam8_dy    =                           4.6500000000e-03
cam8_dpx   =                           4.6500000000e-03
cam8_dpy   =                           4.6500000000e-03
cam8_focal =                           3.6936729025e+00
cam8_kappa1=                           -6.1986350417e-02
cam8_cx    =                           4.3610005997e+02 
cam8_cy    =                           2.5792023557e+02
cam8_sx    =                           1.0000000000e+00
cam8_tx    =                           8.6041235949e+03 
cam8_ty    =                           -4.4715843627e+02 
cam8_tz    =                           1.4593893245e+04 
cam8_rx    =                           1.7745394894e+00 
cam8_ry    =                           3.6370404431e-01
cam8_rz    =                           9.2805843657e-02
   
#cam view5
#cam5_ncx     =1.6000000000e+03
#cam5_nfx     =1.6000000000e+03
#cam5_dx      =4.6500000000e-03
#cam5_dy      =4.6500000000e-03
#cam5_dpx     =4.6500000000e-03
#cam5_dpy     =4.6500000000e-03
#cam5_focal   =3.8593934840e+00
#cam5_kappa1  =2.2757857614e-02
#cam5_cx      =2.4853856155e+02
#cam5_cy      =3.3791011750e+02
#cam5_sx      =1.0000000000e+00
#cam5_tx      =-7.6101258932e+03
#cam5_ty      =-9.8639923333e+02
#cam5_tz      =1.2748530990e+04
#cam5_rx      =-2.1094320856e+00
#cam5_ry      =-1.0807782908e+00
#cam5_rz      =-2.6584356063e+00


#View_006

cam5_ncx   =               1.6000000000e+03
cam5_nfx   =               1.6000000000e+03
cam5_dx    =               4.6500000000e-03
cam5_dy    =               4.6500000000e-03
cam5_dpx   =               4.6500000000e-03
cam5_dpy   =               4.6500000000e-03
cam5_focal =               4.0766744114e+00
cam5_kappa1=               1.8390355585e-02
cam5_cx    =               5.4778587794e+02
cam5_cy    =               2.0025964732e+02
cam5_sx    =               1.0000000000e+00
cam5_tx    =              -9.4185059582e+03  
cam5_ty    =               -6.8342071459e+02 
cam5_tz    =               1.8334013493e+04
cam5_rx    =               2.2181263060e+00
cam5_ry    =              -1.3115850224e+00 
cam5_rz    =             -6.2222518243e-01



#"View_007"
#cam5_ncx   =                           1.6000000000e+03
#cam5_nfx   =                           1.6000000000e+03
#cam5_dx    =                           4.6500000000e-03
#cam5_dy    =                           4.6500000000e-03
#cam5_dpx   =                           4.6500000000e-03
#cam5_dpy   =                           4.6500000000e-03
#cam5_focal =                           3.6936729025e+00
#cam5_kappa1=                           -6.1986350417e-02
#cam5_cx    =                           4.3610005997e+02 
#cam5_cy    =                           2.5792023557e+02
#cam5_sx    =                           1.0000000000e+00
#cam5_tx    =                           8.6041235949e+03 
#cam5_ty    =                           -4.4715843627e+02 
#cam5_tz    =                           1.4593893245e+04 
#cam5_rx    =                           1.7745394894e+00 
#cam5_ry    =                           3.6370404431e-01
#cam5_rz    =                           9.2805843657e-02


import cv2
import numpy as np

dir_path="/media/da/fc9cb2c7-d160-4b66-9178-fe9d57d0c0ce/Crowd_PETS09/S2/L1/Time_12-34/View_006/" ##  collabarative camera View_008

dir_path2="/media/da/fc9cb2c7-d160-4b66-9178-fe9d57d0c0ce/Crowd_PETS09/S2/L1/Time_12-34/View_007/"



count=0

filename = dir_path + "frame_000"+str(0)+".jpg"
filename3 = dir_path + "frame_000"+str(0)+".jpg"

filename2 = dir_path2 + "frame_000"+str(0)+".jpg"
img = cv2.imread(filename)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img3 = cv2.imread(filename3)
img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
img2 = cv2.imread(filename2)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
#image1 = cv2.resize(img,(720,576))
#image1 = np.array(image1,dtype=np.float32
#image2 = cv2.resize(img2,(768,576))
#image3 = cv2.resize(img3,(720,576))

#image2 = np.array(image2,dtype=np.float32)

orig_2 = []  # Store the images here.
input_images2 = []  # Store resized versions of the images here.
orig_2.append(img)
orig_3 = []
orig_3.append(img3)

input_images2.append(img2)


for y_index in range(0,576):
 for x_index in range(0,720):
  cam5_xf=((x_index))
  cam5_yf=y_index
  cv2.rectangle(orig_2[0],(x_index, y_index), (x_index, y_index),(0,255,255),4)
     
  #cam5 is the collob cam
  cam5_yd=(cam5_yf-cam5_cy)*cam5_dy
  cam5_xd=((cam5_xf-cam5_cx)*cam5_dx)/cam5_sx
  #print(cam5_xd,cam5_yd)
  cam5_r=math.sqrt(cam5_xd*cam5_xd+cam5_yd*cam5_yd)
  
  cam5_xu=cam5_xd*(1+cam5_kappa1*cam5_r*cam5_r)
  cam5_yu=cam5_yd*(1+cam5_kappa1*cam5_r*cam5_r)
  
  r1=cos(cam5_ry)*cos(cam5_rz)
  r2=cos(cam5_rz)*sin(cam5_rx)*sin(cam5_ry)-cos(cam5_rx)*sin(cam5_rz)
  r3=sin(cam5_rx)*sin(cam5_rz)+cos(cam5_rx)*cos(cam5_rz)*sin(cam5_ry)
  r4=cos(cam5_ry)*sin(cam5_rz)
  r5=sin(cam5_rx)*sin(cam5_ry)*sin(cam5_rz)+cos(cam5_rx)*cos(cam5_rz)
  r6=cos(cam5_rx)*sin(cam5_ry)*sin(cam5_rz)-cos(cam5_rz)*cos(cam5_rx)
  r7=-sin(cam5_ry)
  r8=cos(cam5_ry)*sin(cam5_rx)
  r9=cos(cam5_rx)*cos(cam5_ry)
  
  
  
  mR11 =r1 
  mR12 =r2  
  mR13 =r3  
  mR21 =r4 
  mR22 =r5  
  mR23 =r6  
  mR31 =r7 
  mR32 =r8 
  mR33 =r9
  
  
  R_cam5=np.zeros((3,3))
  R_cam5[0][0]=r1
  R_cam5[0][1]=r2
  R_cam5[0][2]=r3
  R_cam5[1][0]=r4
  R_cam5[1][1]=r5
  R_cam5[1][2]=r6
  R_cam5[2][0]=r7
  R_cam5[2][1]=r8
  R_cam5[2][2]=r9
  T_cam5=np.zeros((3,1))
  T_cam5[0][0]=cam5_tx
  T_cam5[1][0]=cam5_ty
  T_cam5[2][0]=cam5_tz
  mTx=cam5_tx
  mTy=cam5_ty
  mTz=cam5_tz
  Yu=cam5_yu
  Xu=cam5_xu
  Zw=0
  #Zw=-7125

  #print(Zw,'zw is')
  mFocal=cam5_focal
  common_denominator = ((mR11 * mR32 - mR12 * mR31) * Yu +(mR22 * mR31 - mR21 * mR32) * Xu -mFocal * mR11 * mR22 + mFocal * mR12 * mR21);
  
  Xw = (((mR12 * mR33 - mR13 * mR32) * Yu +
  			(mR23 * mR32 - mR22 * mR33) * Xu -
  			mFocal * mR12 * mR23 + mFocal * mR13 * mR22) * Zw +
  			(mR12 * mTz - mR32 * mTx) * Yu +
  			(mR32 * mTy - mR22 * mTz) * Xu -
  			mFocal * mR12 * mTy + mFocal * mR22 * mTx) / common_denominator
  	
  Yw = -(((mR11 * mR33 - mR13 * mR31) * Yu +
  			(mR23 * mR31 - mR21 * mR33) * Xu -
  			mFocal * mR11 * mR23 + mFocal * mR13 * mR21) * Zw +
  			(mR11 * mTz - mR31 * mTx) * Yu +
  			(mR31 * mTy - mR21 * mTz) * Xu -
  			mFocal * mR11 * mTy + mFocal * mR21 * mTx) / common_denominator
  
  #print('Xw and Yw are',Xw,Yw)
  #Xw=-4285.2738865021365 
  #Xy =-7417.0422861499255

  p1=cos(cam8_ry)*cos(cam8_rz)
  p2=cos(cam8_rz)*sin(cam8_rx)*sin(cam8_ry)-cos(cam8_rx)*sin(cam8_rz)
  p3=sin(cam8_rx)*sin(cam8_rz)+cos(cam8_rx)*cos(cam8_rz)*sin(cam8_ry)
  p4=cos(cam8_ry)*sin(cam8_rz)
  p5=sin(cam8_rx)*sin(cam8_ry)*sin(cam8_rz)+cos(cam8_rx)*cos(cam8_rz)
  p6=cos(cam8_rx)*sin(cam8_ry)*sin(cam8_rz)-cos(cam8_rz)*cos(cam8_rx)
  p7=-sin(cam8_ry)
  p8=cos(cam8_ry)*sin(cam8_rx)
  p9=cos(cam8_rx)*cos(cam8_ry)
  
  R_cam8=np.zeros((3,3))
  R_cam8[0][0]=p1
  R_cam8[0][1]=p2
  R_cam8[0][2]=p3
  R_cam8[1][0]=p4
  R_cam8[1][1]=p5
  R_cam8[1][2]=p6
  R_cam8[2][0]=p7
  R_cam8[2][1]=p8
  R_cam8[2][2]=p9
  T_cam8=np.zeros((3,1))
  T_cam8[0][0]=cam8_tx
  T_cam8[1][0]=cam8_ty
  T_cam8[2][0]=cam8_tz
  world_co=np.zeros((3,1))
  world_co[0][0]=Xw
  world_co[1][0]=Yw
  world_co[2][0]=Zw
  cam8_code=np.dot(R_cam8,world_co)+T_cam8
  #print(cam8_code)
  cam8_Xi=cam8_code[0][0]
  cam8_Yi=cam8_code[1][0]
  cam8_Zi=cam8_code[2][0]
  cam8_xu=(cam8_Xi*cam8_focal)/cam8_Zi
  cam8_yu=(cam8_Yi*cam8_focal)/cam8_Zi
  
  
  #print(cam8_xu,cam8_yu,'kkkkkkkkkk')
  Ru = sqrt(cam8_xu*cam8_xu + cam8_yu*cam8_yu)
  c = 1.0 / cam8_kappa1
  d = -c * Ru
  
  Q = c / 3
  R = -d / 2
  D = Q*Q*Q + R*R
  
  if (D >= 0): 
  	#/* one real root */
  	D = sqrt(D);
  	if (R + D > 0):
  		S = pow(R + D, 1.0/3.0)
  	else:
  		S = -pow(-R - D, 1.0/3.0)
  	if (R - D > 0):
  		T = pow(R - D, 1.0/3.0)
  	else:
  	
  		T = -pow(D - R, 1.0/3.0)
  	
  	Rd = S + T
  	
  	if (Rd < 0): 
  		Rd = sqrt(-1.0 / (3 * cam8_kappa1))
  		#/*fprintf (stderr, "\nWarning: undistorted image point to distorted image point mapping limited by\n");
  		#fprintf (stderr, "         maximum barrel distortion radius of %lf\n", Rd);
  		#fprintf (stderr, "         (Xu = %lf, Yu = %lf) -> (Xd = %lf, Yd = %lf)\n\n", Xu, Yu, Xu * Rd / Ru, Yu * Rd / Ru);*/
  else:
  	#/* three real roots */
    D = sqrt(-D)
    
  
    S = pow( sqrt(R*R + D*D) , 1.0/3.0 )
    #print('came hereeeeeeeeeeeee',D,R,S)
    T = atan2(D,R) / 3
    #print(T)
    sinT = sin(T)
    cosT = cos(T)
    
    #/* the larger positive root is    2*S*cos(T)                   */
    #/* the smaller positive root is   -S*cos(T) + SQRT(3)*S*sin(T) */
    #/* the negative root is           -S*cos(T) - SQRT(3)*S*sin(T) */
    
    Rd = -S * cos(T) + sqrt(3.0) * S * sin(T)#	/* use the smaller positive root */
    #print('Rd isss',Rd)
  lambda1 = Rd / Ru;
  
  cam8_xd = cam8_xu * lambda1;
  cam8_yd = cam8_yu * lambda1;
  
  #print(cam8_xd,cam8_yd)
  cam8_xf=int(((cam8_sx*cam8_xd)/cam8_dx+cam8_cx))
  cam8_yf=int((cam8_yd)/cam8_dy+cam8_cy)
  
  
  
  
  
  
  #print('calculated x and ya are',cam8_xf,cam8_yf,'true x and y are ',cam8_re_xf,cam8_re_yf)
  
  cv2.rectangle(input_images2[0],(cam8_xf, cam8_yf), (cam8_xf, cam8_yf),(255,255,255),25)
  if(0<cam8_xf<720 and 0<cam8_yf<576):
   count+=1

#im = cv.imread('test.jpg')
#imgray = cv2.cvtColor(input_images2[0],cv2.COLOR_BGR2GRAY)
#ret, thresh = cv2.threshold(imgray, 127, 255, 0)
#contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
##print(contours,hierarchy)
#a=0
#maxn=0
#for i in contours:
#    current_maxn=len(i)
#    if(current_maxn>maxn):
#         maxn=current_maxn
#         max_index=a
#    a+=1
#    cv2.drawContours(input_images2[0], contours, a, (0,255,0), 3)
#for x in range(677):
# m=(-224+242)/(677-0)
# c=-242
# y=int(abs(m*x+c))
# cv2.rectangle(input_images2[0],(x,y ), (x, y),(255,0,0),2)
#for x in range(677,720):
# m=(-224+241)/(720-677)
# c=-45
# y=int(abs(m*x+c))
# cv2.rectangle(input_images2[0],(x,y ), (x, y),(255,0,0),2)
imgray = cv2.cvtColor(input_images2[0], cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 254, 255, 0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
c = max(contours, key = cv2.contourArea)
#print('overlapping coodinate for cam8 is',c)
cv2.drawContours(input_images2[0], [c], -1, (0,255,0), 3)

#print('percentage area of overlap from view6 on view8 is',count/(720*576)*100,"%")
#cv2.imshow('collob cam5',orig_2[0])
#cv2.imshow('collob cam5_initial',orig_3[0])

cv2.imshow('referance cam7',input_images2[0])
cv2.waitKey(1)
f = open("overlapping_coordscam7cam6.txt", "a")
string=" overlapping coords for cam7 is {} \n".format(c)  
#print(string)
f.write(string)
f.close()
#cv2.imshow('edge',edges)
for y_index in range(0,576):
 for x_index in range(0,720):
  cam8_xf=((x_index))
  cam8_yf=y_index
  cv2.rectangle(orig_2[0],(x_index, y_index), (x_index, y_index),(0,255,255),4)
     
  #cam8 is the collob cam
  cam8_yd=(cam8_yf-cam8_cy)*cam8_dy
  cam8_xd=((cam8_xf-cam8_cx)*cam8_dx)/cam8_sx
  #print(cam8_xd,cam8_yd)
  cam8_r=math.sqrt(cam8_xd*cam8_xd+cam8_yd*cam8_yd)
  
  cam8_xu=cam8_xd*(1+cam8_kappa1*cam8_r*cam8_r)
  cam8_yu=cam8_yd*(1+cam8_kappa1*cam8_r*cam8_r)
  
  r1=cos(cam8_ry)*cos(cam8_rz)
  r2=cos(cam8_rz)*sin(cam8_rx)*sin(cam8_ry)-cos(cam8_rx)*sin(cam8_rz)
  r3=sin(cam8_rx)*sin(cam8_rz)+cos(cam8_rx)*cos(cam8_rz)*sin(cam8_ry)
  r4=cos(cam8_ry)*sin(cam8_rz)
  r5=sin(cam8_rx)*sin(cam8_ry)*sin(cam8_rz)+cos(cam8_rx)*cos(cam8_rz)
  r6=cos(cam8_rx)*sin(cam8_ry)*sin(cam8_rz)-cos(cam8_rz)*cos(cam8_rx)
  r7=-sin(cam8_ry)
  r8=cos(cam8_ry)*sin(cam8_rx)
  r9=cos(cam8_rx)*cos(cam8_ry)
  
  
  
  mR11 =r1 
  mR12 =r2  
  mR13 =r3  
  mR21 =r4 
  mR22 =r5  
  mR23 =r6  
  mR31 =r7 
  mR32 =r8 
  mR33 =r9
  
  
  R_cam8=np.zeros((3,3))
  R_cam8[0][0]=r1
  R_cam8[0][1]=r2
  R_cam8[0][2]=r3
  R_cam8[1][0]=r4
  R_cam8[1][1]=r5
  R_cam8[1][2]=r6
  R_cam8[2][0]=r7
  R_cam8[2][1]=r8
  R_cam8[2][2]=r9
  T_cam8=np.zeros((3,1))
  T_cam8[0][0]=cam8_tx
  T_cam8[1][0]=cam8_ty
  T_cam8[2][0]=cam8_tz
  mTx=cam8_tx
  mTy=cam8_ty
  mTz=cam8_tz
  Yu=cam8_yu
  Xu=cam8_xu
  Zw=0
  #Zw=-7125

  #print(Zw,'zw is')
  mFocal=cam8_focal
  common_denominator = ((mR11 * mR32 - mR12 * mR31) * Yu +(mR22 * mR31 - mR21 * mR32) * Xu -mFocal * mR11 * mR22 + mFocal * mR12 * mR21);
  
  Xw = (((mR12 * mR33 - mR13 * mR32) * Yu +
  			(mR23 * mR32 - mR22 * mR33) * Xu -
  			mFocal * mR12 * mR23 + mFocal * mR13 * mR22) * Zw +
  			(mR12 * mTz - mR32 * mTx) * Yu +
  			(mR32 * mTy - mR22 * mTz) * Xu -
  			mFocal * mR12 * mTy + mFocal * mR22 * mTx) / common_denominator
  	
  Yw = -(((mR11 * mR33 - mR13 * mR31) * Yu +
  			(mR23 * mR31 - mR21 * mR33) * Xu -
  			mFocal * mR11 * mR23 + mFocal * mR13 * mR21) * Zw +
  			(mR11 * mTz - mR31 * mTx) * Yu +
  			(mR31 * mTy - mR21 * mTz) * Xu -
  			mFocal * mR11 * mTy + mFocal * mR21 * mTx) / common_denominator
  
  #print('Xw and Yw are',Xw,Yw)
  #Xw=-4285.2738865021365 
  #Xy =-7417.0422861499255

  p1=cos(cam5_ry)*cos(cam5_rz)
  p2=cos(cam5_rz)*sin(cam5_rx)*sin(cam5_ry)-cos(cam5_rx)*sin(cam5_rz)
  p3=sin(cam5_rx)*sin(cam5_rz)+cos(cam5_rx)*cos(cam5_rz)*sin(cam5_ry)
  p4=cos(cam5_ry)*sin(cam5_rz)
  p5=sin(cam5_rx)*sin(cam5_ry)*sin(cam5_rz)+cos(cam5_rx)*cos(cam5_rz)
  p6=cos(cam5_rx)*sin(cam5_ry)*sin(cam5_rz)-cos(cam5_rz)*cos(cam5_rx)
  p7=-sin(cam5_ry)
  p8=cos(cam5_ry)*sin(cam5_rx)
  p9=cos(cam5_rx)*cos(cam5_ry)
  
  R_cam5=np.zeros((3,3))
  R_cam5[0][0]=p1
  R_cam5[0][1]=p2
  R_cam5[0][2]=p3
  R_cam5[1][0]=p4
  R_cam5[1][1]=p5
  R_cam5[1][2]=p6
  R_cam5[2][0]=p7
  R_cam5[2][1]=p8
  R_cam5[2][2]=p9
  T_cam5=np.zeros((3,1))
  T_cam5[0][0]=cam5_tx
  T_cam5[1][0]=cam5_ty
  T_cam5[2][0]=cam5_tz
  world_co=np.zeros((3,1))
  world_co[0][0]=Xw
  world_co[1][0]=Yw
  world_co[2][0]=Zw
  cam5_code=np.dot(R_cam5,world_co)+T_cam5
  #print(cam5_code)
  cam5_Xi=cam5_code[0][0]
  cam5_Yi=cam5_code[1][0]
  cam5_Zi=cam5_code[2][0]
  cam5_xu=(cam5_Xi*cam5_focal)/cam5_Zi
  cam5_yu=(cam5_Yi*cam5_focal)/cam5_Zi
  
  
  #print(cam5_xu,cam5_yu,'kkkkkkkkkk')
  Ru = sqrt(cam5_xu*cam5_xu + cam5_yu*cam5_yu)
  c = 1.0 / cam5_kappa1
  d = -c * Ru
  
  Q = c / 3
  R = -d / 2
  D = Q*Q*Q + R*R
  
  if (D >= 0): 
  	#/* one real root */
  	D = sqrt(D);
  	if (R + D > 0):
  		S = pow(R + D, 1.0/3.0)
  	else:
  		S = -pow(-R - D, 1.0/3.0)
  	if (R - D > 0):
  		T = pow(R - D, 1.0/3.0)
  	else:
  	
  		T = -pow(D - R, 1.0/3.0)
  	
  	Rd = S + T
  	
  	if (Rd < 0): 
  		Rd = sqrt(-1.0 / (3 * cam5_kappa1))
  		#/*fprintf (stderr, "\nWarning: undistorted image point to distorted image point mapping limited by\n");
  		#fprintf (stderr, "         maximum barrel distortion radius of %lf\n", Rd);
  		#fprintf (stderr, "         (Xu = %lf, Yu = %lf) -> (Xd = %lf, Yd = %lf)\n\n", Xu, Yu, Xu * Rd / Ru, Yu * Rd / Ru);*/
  else:
  	#/* three real roots */
    D = sqrt(-D)
    
  
    S = pow( sqrt(R*R + D*D) , 1.0/3.0 )
    #print('came hereeeeeeeeeeeee',D,R,S)
    T = atan2(D,R) / 3
    #print(T)
    sinT = sin(T)
    cosT = cos(T)
    
    #/* the larger positive root is    2*S*cos(T)                   */
    #/* the smaller positive root is   -S*cos(T) + SQRT(3)*S*sin(T) */
    #/* the negative root is           -S*cos(T) - SQRT(3)*S*sin(T) */
    
    Rd = -S * cos(T) + sqrt(3.0) * S * sin(T)#	/* use the smaller positive root */
    #print('Rd isss',Rd)
  lambda1 = Rd / Ru;
  
  cam5_xd = cam5_xu * lambda1;
  cam5_yd = cam5_yu * lambda1;
  
  #print(cam5_xd,cam5_yd)
  cam5_xf=int(((cam5_sx*cam5_xd)/cam5_dx+cam5_cx))
  cam5_yf=int((cam5_yd)/cam5_dy+cam5_cy)
  
  
  
  
  
  
  #print('calculated x and ya are',cam5_xf,cam5_yf,'true x and y are ',cam5_re_xf,cam5_re_yf)
  
  cv2.rectangle(orig_3[0],(cam5_xf, cam5_yf), (cam5_xf, cam5_yf),(255,255,255),25)
  if(0<cam5_xf<720 and 0<cam5_yf<576):
   count+=1


imgray = cv2.cvtColor(orig_3[0], cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(imgray, 254, 255, 0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
c = max(contours, key = cv2.contourArea)
#print('overlapping coodinate for cam8 is',c)
cv2.drawContours(orig_3[0], [c], -1, (0,255,0), 3)

cv2.imshow('collab cam6',orig_3[0])
cv2.waitKey(0)
f = open("overlapping_coordscam7cam6.txt", "a")
string=" overlapping coords for cam6 is {} \n".format(c)  
#print(string)
f.write(string)
f.close()
