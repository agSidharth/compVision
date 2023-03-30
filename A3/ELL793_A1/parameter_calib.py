import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


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



def DLT(D_norm,D_trans,d_norm,d_trans):
    n = D_norm.shape[0]
    O = np.array([0,0,0,0])
    Q = np.zeros((2*n,12))
    for i in range(n):
        temp1 = np.array([])
        temp1 = np.append(temp1, D_norm[i,:])
        temp1 = np.append(temp1, O)
        temp1 = np.append(temp1, -d_norm[i,0]*D_norm[i,:])
        Q[2*i, :] = temp1
        temp2 = np.array([])
        temp2 = np.append(temp2, O)
        temp2 = np.append(temp2, D_norm[i,:])
        temp2 = np.append(temp2, -d_norm[i,1]*D_norm[i,:])
        Q[2*i+1, :] = temp2
    A = np.matmul(Q.T, Q)
    w, v = np.linalg.eig(A)
    idx = np.argmin(w)
    vec = v[:,idx]
    M_norm = np.reshape(vec, (3,4))
    M = np.matmul(np.matmul(np.linalg.inv(d_trans), M_norm), D_trans)
    return M



def extract_parameters(M):

    A = M[:,0:3]
    R = np.zeros((3,3))
    K = np.zeros((3,3))
    rho = -1/np.linalg.norm(A[2,:])
    R[2,:] = rho*A[2,:]
    X0 = rho*rho*(np.matmul(A[0,:], A[2,:].T))
    Y0 = rho*rho*(np.matmul(A[1,:], A[2,:].T))
    cross1 = np.cross(A[0,:],A[2,:])
    cross2 = np.cross(A[1,:],A[2,:])
    n_cross1 = np.linalg.norm(cross1)
    n_cross2 = np.linalg.norm(cross2)
    c1 = -cross1/n_cross1
    c2 = cross2[...,None]/n_cross2
    theta = np.arccos(np.matmul(c1, c2))
    alpha = rho*rho*n_cross1*np.sin(theta)
    beta = rho*rho*n_cross2*np.sin(theta)

    R[0,:] = cross2/n_cross2
    R[1,:] = np.cross(R[2,:],R[0,:])
    print("R =\n", R)

    K[0,:] = np.array([alpha, -alpha/np.tan(theta), X0])
    K[1,:] = np.array([0, beta/np.sin(theta), Y0])
    K[2,:] = np.array([0, 0, 1])
    print("K =\n", K)

    u0 = K[0,2]
    v0 = K[1,2]
    print("u0 =", u0)
    print("v0 =", v0)
    print("theta =", theta)
    print("alpha =", alpha)
    print("beta =", beta)

    t = (np.matmul(rho*np.linalg.inv(K), M[:,3].T))[...,None]
    print("t=\n", t)

    x0 = -np.matmul(np.linalg.inv(R), t)
    print("x0=\n", x0)

if __name__ == '__main__':
    ds = pd.read_csv("coord_data.csv")
    w_c = (ds[ds.columns[1:4]]).to_numpy()
    p_c = (ds[ds.columns[4:]]).to_numpy()


    n = 30 

    w = w_c[:n,:]
    p = p_c[:n,:]
    p_norm,p_trans = normalise(p)
    w_norm,w_trans = normalise(w)

    # Convert to Homogenous Coordinates
    w = np.append(w, np.ones((n,1)), axis=1)
    p = np.append(p, np.ones((n,1)), axis=1)



    M = DLT(w_norm,w_trans,p_norm,p_trans)
    print(M)

    extract_parameters(M) 

