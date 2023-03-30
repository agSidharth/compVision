import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import sys
import scipy
import math

NORMALIZE = False
TEST = True

class Calibrator:
    def __init__(self,fileName):

        df = pd.read_csv(fileName)
        self.n = len(df.index)

        threeD_cols = df.columns[0:3]
        twoD_cols = df.columns[3:5]
        self.w = df[threeD_cols].to_numpy()
        self.p = df[twoD_cols].to_numpy()

        if TEST:
            perm = np.random.permutation(range(self.n))
            self.p_test = np.append(self.p[perm],np.ones((self.n,1)),axis=1)
            self.w_test = np.append(self.w[perm],np.ones((self.n,1)),axis=1)

        self.calibrate()

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
        
        return M_temp/M_temp[2,3]


    def extract_params(self):
        self.K,R =  scipy.linalg.rq(self.M[:,0:3]) 
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
            self.M = np.matmul(np.matmul(np.linalg.inv(p_trans),M_temp),w_trans)
        else:
            self.p = np.append(self.p,np.ones((self.n,1)),axis = 1)
            self.w = np.append(self.w,np.ones((self.n,1)),axis = 1)
            self.M = self.DLT(self.p,self.w)
        self.extract_params()
        self.printparams()
    
    def test(self):
        pts = np.matmul(self.M,self.w_test.T).T
        pts[:,0] = pts[:,0]/pts[:,2]
        pts[:,1] = pts[:,1]/pts[:,2]
        pts[:,2] = np.ones((self.n))

        rmse = np.linalg.norm(pts-self.p_test)/np.sqrt(self.n)
        print("The rmse error is : "+str(rmse))

        plt.scatter(self.p_test[:,0],self.p_test[:,1],marker = 'x')
        plt.scatter(pts[:,0],pts[:,1],marker = 'o')
        plt.xlabel('X cord')
        plt.ylabel('Y cord')
        plt.title('Visualizing of performance of camera calibration')
        plt.legend(['original','predicted'])
        plt.savefig('output.png')
        pass
        

if __name__=='__main__':
    ourCalib = Calibrator(sys.argv[1])
    if TEST: ourCalib.test()
    

    




