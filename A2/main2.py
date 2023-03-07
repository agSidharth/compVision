import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
import random

SHOW_CORN = False
SHOW_PATCH = False
SSD_THRESH = 2200
DESC_SIZE = 5

def checkCorner(corner,imgShape, delta = DESC_SIZE):
    if corner[0]-delta<0 or corner[0]+delta>=imgShape[0]: return False
    if corner[1]-delta<0 or corner[1]+delta>=imgShape[1]: return False
    return True


def get_descriptors(corners,image_name,delta = DESC_SIZE):
    img = cv2.imread(image_name)
    descriptors = []
    for c in corners:
        patch = img[c[0] - delta : c[0] + delta, c[1] - delta : c[1] + delta]
        if not checkCorner(c,img.shape):
            patch = np.zeros((2*delta,2*delta,3))
        descriptors.append(patch.flatten())
    return np.array(descriptors,dtype = 'int')

def visualize_patch(corner,image_name, delta = 100):
    img_copy = cv2.imread(image_name)

    img_copy = cv2.circle(img_copy,(corner[1],corner[0]),8,[0,255,255],2)
    #img_copy[corner[0],corner[1]] = [0,255,255]

    patch = img_copy[corner[0] - delta : corner[0] + delta, corner[1] - delta : corner[1] + delta]
    
    cv2.imshow("PATCH_"+image_name+str(corner),cv2.resize(patch,(400,400),cv2.INTER_LINEAR))

def Hessian_Corner_Detector(image_name):

    image_input = cv2.imread(image_name)

    image_input_gray = cv2.GaussianBlur(cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY),(9,9),3)

    Ix = cv2.Sobel(image_input_gray, cv2.CV_64F, 1, 0, ksize=9) 
    Iy = cv2.Sobel(image_input_gray, cv2.CV_64F, 0, 1, ksize=9)

    Ix_2 = cv2.GaussianBlur(np.multiply(Ix, Ix),(9,9),3) 
    Iy_2 = cv2.GaussianBlur(np.multiply(Iy, Iy),(9,9),3)  
    IxIy = cv2.GaussianBlur(np.multiply(Ix, Iy),(9,9),3)   

    k = 0.05
    R = np.zeros(image_input_gray.shape)*k + np.multiply(Ix_2,Iy_2)-np.square(IxIy)-k*np.square(Ix_2+Iy_2)
    
    dst_thresh = cv2.dilate(cv2.threshold(R,0.05*R.max(),255,cv2.THRESH_BINARY)[1],None)
    outImg = image_input.copy()
    #outImg[dst_thresh==255] = [0,255,255]

    corn_x,corn_y = np.where(dst_thresh==255)
    corners = np.array([[corn_x[idx],corn_y[idx]] for idx in range(len(corn_x))])

    for c in corners:
        outImg = cv2.circle(outImg,(c[1],c[0]),5,[0,255,255],3)
    
    if SHOW_CORN:
        cv2.imshow(image_name[:-4]+"_corners.jpg",outImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return corners

if __name__ == "__main__":

    inputs = os.listdir(sys.argv[1])
    inputs.sort()

    if "image " in inputs[0]:
        indexes = [int(input[6:-4]) for input in inputs]
        indexes.sort()
        inputs = ["image "+str(idx)+".jpg" for idx in indexes]
        pass
    
    inputs = [os.path.join(sys.argv[1],idx) for idx in inputs]

    for kdx in range(0,len(inputs)-1):
        corners1 = Hessian_Corner_Detector(inputs[kdx])
        corners2 = Hessian_Corner_Detector(inputs[kdx+1])

        desc1 = get_descriptors(corners1,inputs[kdx])
        desc2 = get_descriptors(corners2,inputs[kdx+1])

        imgShape1 = cv2.imread(inputs[kdx]).shape[:2]
        imgShape2 = cv2.imread(inputs[kdx+1]).shape[:2]

        matches = []
        for idx in tqdm(range(len(corners1))):
            c1 = corners1[idx]

            if not checkCorner(c1,imgShape1): continue

            d1 = desc1[idx]

            min_dist = np.inf
            min_jdx = -1

            for jdx in range(len(corners2)):
                c2 = corners2[jdx]
                if not checkCorner(c2,imgShape2): continue

                thisDist = abs(c1[0]-c2[0]) + abs(c1[1]-c2[1])
                if thisDist<min_dist:
                    min_dist = thisDist
                    min_jdx = jdx
            
            c2 = corners2[min_jdx]
            d2 = desc2[min_jdx]

            if(d1.shape!=d2.shape):
                print(kdx)
                print(d1.shape)
                print(corners1[idx])
                print(desc1[idx])
            ssd = np.linalg.norm(d1-d2)

            if ssd<SSD_THRESH:
                matches.append([idx,min_jdx])
            
            #if num_matches>0: break 

        if SHOW_PATCH:
            rand_index = random.randint(0,len(matches)-1)
            print(rand_index)
            visualize_patch(corners1[matches[rand_index][0]],inputs[kdx])
            visualize_patch(corners2[matches[rand_index][1]],inputs[kdx+1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print("Number of corners in "+inputs[kdx]+" : "+str(len(corners1)))
        print("Number of corners in "+inputs[kdx+1]+" : "+str(len(corners2)))
        print("Number of matches found : "+str(len(matches))+"\n")
        






    