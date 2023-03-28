import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import parameter_calib

def normalise(D):
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




def test(M,D,d,path=None):
    if D.shape[0]==0:
        return
    projected_pts = (np.matmul(M, D.T)).T
    projected_pts[:,0] = projected_pts[:,0]/(projected_pts[:,2])
    projected_pts[:,1] = projected_pts[:,1]/(projected_pts[:,2])
    projected_pts[:,2] = projected_pts[:,2]/(projected_pts[:,2])

    err = projected_pts - d
    sqe = err**2
    mse = np.mean(sqe[:,:2])
    rmse = np.sqrt(mse)
    print("Error is " ,rmse)

    if path is not None:
        plt.axis('off')
        plt.figure(figsize=(40,40))
        plt.scatter(d[:,0],d[:,1],facecolors='none',edgecolor='c',marker='X',linewidth=10,s=2000)

        plt.scatter(projected_pts[:,0],projected_pts[:,1],color='m',marker='o',s=2000)
        plt.xlabel('X',fontsize=100)
        plt.ylabel("Y",fontsize=100)
        plt.legend(['Actual Points','Predicted Points'],loc='lower left',prop={'size':100},fontsize=100)
        plt.grid(linewidth=10)
        plt.savefig(path)

    return

if __name__ == '__main__':

    ds = pd.read_csv("coord_data.csv")

    w_c = (ds[ds.columns[1:4]]).to_numpy()
    p_c = (ds[ds.columns[4:]]).to_numpy()


    # Train Test Split
    n = 30 # Select number of training data points
    w_test = w_c[-n:,:]
    p_test = p_c[-n:,:]

    w = w_c[:n,:]
    p = p_c[:n,:]
    p_norm,p_trans = parameter_calib.normalise(p)
    w_norm,w_trans = parameter_calib.normalise(w)


    w = np.append(w, np.ones((n,1)), axis=1)
    p = np.append(p, np.ones((n,1)), axis=1)



    M = parameter_calib.DLT(w_norm,w_trans,p_norm,p_trans)

    p_test = np.append(p_test, np.ones((len(p_test),1)), axis=1)
    w_test = np.append(w_test, np.ones((len(w_test),1)), axis=1)

    test(M,w_test,p_test,'final_proj.png'.format(n))


