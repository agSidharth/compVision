import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import sys
import scipy
import math

NORMALIZE = True

class Calibrator:
    def __init__(self,fileName):

        df = pd.read_csv(fileName)
        self.n = len(df.index)

        threeD_cols = df.columns[1:4]
        twoD_cols = df.columns[4:6]
        self.w = df[threeD_cols].to_numpy()
        self.p = df[twoD_cols].to_numpy()

        self.calibrate()

    """
    def normalise2(self,D):
        dim = D.shape[1]
        n = D.shape[0]
        
        D_centre = [sum(x)/len(x) for x in zip(*D)]
        D_centre = np.asarray(D_centre)
        D0 = D - D_centre

        avg_dist_D0 = 0
        for i in range(n):
            avg_dist_D0 = avg_dist_D0 + np.linalg.norm(D0[i,:])
        avg_dist_D0 = avg_dist_D0/float(n)
        scaler1 = np.sqrt(dim)/float(avg_dist_D0)
        D_norm = scaler1*D0
        D_norm = np.append(D_norm, np.ones((n,1)), axis=1)
        D_scale = np.diag([*[scaler1 for i in range(0,dim)],1.0])
        D_shift = np.eye(dim)
        D_shift = np.append(D_shift, [[0]*dim], axis=0)
        D_aux1 = -D_centre
        D_aux2 = np.append(D_aux1, [1])
        D_aux3 = D_aux2[..., None]
        D_shift = np.append(D_shift, D_aux3, axis=1)
        D_trans = np.matmul(D_scale, D_shift) 
        return D_norm,D_trans
    """
    
    # see if this needs to be done or not...
    def normalise(self,D):
        dim = D.shape[1]

        D_centre = []
        for d in range(dim):
            D_centre.append(sum(D[:,d])/len(D[:,d]))
        
        D0 = D - np.asarray(D_centre)
        norm_vector = np.linalg.norm(D0,axis = 1)
        scaler = np.sqrt(dim)/(np.sum(norm_vector)/self.n)

        D_norm = np.append(D0*scaler,np.ones((self.n,1)),axis = 1)

        # complete this function
        D_trans = None

        return D_norm,D_trans

    def DLT(self,d_norm,D_norm):
        Q = np.zeros((2*self.n,12))

        for i in range(self.n):
            Q[2*i,:4] = D_norm[i,:]
            Q[2*i,8:] = -1*d_norm[i,0]*D_norm[i,:]
            Q[2*i+1,4:8] = D_norm[i,:]
            Q[2*i+1,8:] = -1*d_norm[i,1]*D_norm[i,:]
        
        _,_,V = np.linalg.svd(Q)
        M_temp = V[-1,:12].reshape((3,4))
        M_temp = M_temp/M_temp[2,3]
        
        return M_temp


    def extract_params(self,M):
        self.K,R =  scipy.linalg.rq(M[:,0:3]) 
        self.K = self.K/self.K[2,2]
        self.K = self.K*np.diag(np.sign(self.K))

        self.alpha = self.K[0,0]
        tanTheta = -self.alpha/self.K[0,1]
        self.theta = math.atan(tanTheta)
        if self.theta<0:
            self.theta += np.pi
        self.x0 = self.K[0,2]
        self.y0 = self.K[1,2]
        self.beta = self.K[1,1]*math.sin(self.theta)
        self.theta = math.degrees(self.theta)
        pass

    def printparams(self):
        print("The K matrix is as follows : ")
        print(self.K)
        print("The intrinsic parameters obtained are as follows : ")
        print("Alpha : "+str(self.alpha))
        print("Beta : "+str(self.beta))
        print("Theta : "+str(self.theta)+" degrees")
        print("x0 : "+str(self.x0))
        print("y0 : "+str(self.y0))

    def calibrate(self):

        if NORMALIZE:
            p_norm,p_trans = self.normalise(self.p)
            w_norm,w_trans = self.normalise(self.w)
            M_temp = self.DLT(p_norm,w_norm)
            M = np.matmul(np.matmul(np.linalg.inv(p_trans),M_temp),w_trans)
        else:
            self.p = np.append(self.p,np.ones((self.n,1)),axis = 1)
            self.w = np.append(self.w,np.ones((self.n,1)),axis = 1)
            M = self.DLT(self.p,self.w)
        self.extract_params(M)
        self.printparams()


if __name__=='__main__':
    ourCalib = Calibrator(sys.argv[1])
    

    




