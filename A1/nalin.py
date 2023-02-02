# %%
import numpy as np
import cv2
import sys

# %%
print("running :",sys.argv[1])
cap = cv2.VideoCapture(sys.argv[1])
ret, frame = cap.read()
frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
if(not ret):
    print("ERROR: Couldnt read first frame")
rows, cols, _ = frame.shape
print(rows, cols)

# %%

gaussians = 3
means = np.random.rand(rows, cols, gaussians)*255
# means[:,:,1] = frame_gray
std = np.full((rows, cols, gaussians) , 5)
# weights = np.zeros((rows, cols, gaussians))
# weights[:,:,0],weights[:,:,1],weights[:,:,2] = 0, 0, 1
# alpha = 0.8
# t = 0.5

# %%
n = 0
while(cap.isOpened()):
    n += 1
    _,frame = cap.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_gray = np.array(frame_gray)
    cv2.imshow("original", frame_gray)

    #check for fg or bg
    mdist = np.array([(abs(means[:,:,i] - frame_gray)) for i in range(gaussians)]) # distance from mean
    minmdist = np.argmin(mdist, axis = 0) #min mdist b/w gaussians
    mgauss = np.take_along_axis(mdist, np.expand_dims(minmdist, axis= 0), axis = 0) #get that mdist
    minstd = np.take_along_axis(std, np.expand_dims(minmdist, axis=-1), axis=-1) #get that std
    vgauss = np.array((mgauss[0,:,:] < 2.5 * minstd[:,:,0]), dtype=np.uint8) #check if background 

    #print foreground
    cv2.imshow('bgsub', (1-vgauss) * 255)

    #update means and stds
    bgs = np.multiply((vgauss), minmdist)
    umeans = np.take_along_axis(means, np.expand_dims(bgs, axis=-1), axis=-1)[:,:,0] #get means to update
    numeans = (vgauss*(frame_gray)) + ((1-vgauss)*(((n-1)*umeans + frame_gray)/n)) #updated means
    np.put_along_axis(means, np.expand_dims(bgs, axis=-1), np.expand_dims(numeans, axis=-1) ,axis=-1)

    ustd = np.take_along_axis(std, np.expand_dims(bgs, axis=-1), axis=-1)[:,:,0] #get std to update
    nustd = (vgauss*400) + ((1-vgauss)*np.sqrt(((n-1)*np.square(ustd)+ np.square(frame_gray - umeans))/n)) #updated means
    np.put_along_axis(std, np.expand_dims(bgs, axis=-1), np.expand_dims(nustd, axis=-1) ,axis=-1)

    reorder = np.argsort(-std, axis = -1)
    means = np.take_along_axis(means,reorder,axis=-1)
    std = np.take_along_axis(std,reorder,axis=-1)

    #print for 424,193
    j = 234
    i = 145
    print(vgauss[i,j], means[i,j], std[i,j], mdist[:, i,j], bgs[i,j])


    if cv2.waitKey(30) & 0xFF == 27:
        break

# %