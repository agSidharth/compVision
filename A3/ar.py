import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import scipy
import cv2 as cv

class ArModel:
    def __init__(self,fileName,s = 1):
        self.w,self.p = self.retCordinates(fileName)
        self.img = cv.imread(fileName[:-3]+"jpeg")
        self.n = self.w.shape[0]
        self.s = s
        self.cube = np.array([[2,2,0,1],[2+s,2,0,1],[2,2+s,0,1],[2+s,2+s,0,1],[2,2,s,1],[2+s,2,s,1],[2,2+s,s,1],[2+s,2+s,s,1]]).T
        self.calibrate()

    def retCordinates(self,fileName):
        df = pd.read_csv(fileName)
        threeD_cols = df.columns[0:2]           #note here z column is always zero
        twoD_cols = df.columns[3:5]
        return df[threeD_cols].to_numpy(), df[twoD_cols].to_numpy()

    def DLT(self,d_norm,D_norm):
        Q = np.zeros((2*self.n,9))

        for i in range(self.n):
            Q[2*i,:3] = D_norm[i,:]
            Q[2*i,6:] = -1*d_norm[i,0]*D_norm[i,:]
            Q[2*i+1,3:6] = D_norm[i,:]
            Q[2*i+1,6:] = -1*d_norm[i,1]*D_norm[i,:]
        
        _,_,V = np.linalg.svd(Q)
        M_temp = V[-1,:12].reshape((3,3))
        M_temp = M_temp/M_temp[2,2]

        Mf = np.zeros((3,4))
        Mf[:,0] = M_temp[:,0].copy()
        Mf[:,1] = M_temp[:,1].copy()
        Mf[:,3] = M_temp[:,2].copy()
        
        return Mf

    def calibrate(self):

        self.p = np.append(self.p,np.ones((self.n,1)),axis = 1)
        self.w = np.append(self.w,np.ones((self.n,1)),axis = 1)
        self.K = np.load('cameraK.npy')
        self.Mt = self.DLT(self.p,self.w)

        Rt = np.matmul(np.linalg.inv(self.K),self.Mt)
        Rt[:,2] = np.cross(Rt[:,0].T,Rt[:,1].T).T
        factor = np.linalg.norm(Rt[:,0])/np.linalg.norm(Rt[:,2])
        Rt[:,2] = Rt[:,2]*factor
        
        self.M = np.matmul(self.K,Rt)

        tempCube = np.matmul(self.M,self.cube)
        arcubef = np.zeros((2,8))
        arcubef[0,:] = tempCube[0,:]/tempCube[2,:]
        arcubef[1,:] = tempCube[1,:]/tempCube[2,:]
        arcubef = arcubef.T.astype(np.int32)

        pairs = [[0,1],[0,2],[1,3],[2,3],[0,4],[1,5],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]]
        for p in pairs:
            self.img = cv.line(self.img,arcubef[p[0]],arcubef[p[1]],(0,255,0),5)

        cv.imwrite("cube.png",self.img)

if __name__=='__main__':
    if(len(sys.argv)==2): armodel = ArModel(sys.argv[1])
    else : armodel = ArModel(sys.argv[1],int(sys.argv[2]))
