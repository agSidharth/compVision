import numpy as np
import cv2 as cv
import os
from numpy.linalg import r, inv
from scipy.stats import multivariate_normal as mv_norm

init_weight = [0.7, 0.11, 0.1, 0.09]
init_u = np.zeros(3)
# initial Covariance matrix
init_sigma = 225*np.eye(3)
init_alpha = 0.05



class GMM():
    def __init__(self, data_dir, train_num, alpha=init_alpha):
        self.data_dir = data_dir
        self.train_num = train_num
        self.alpha = alpha
        self.img_shape = None

        self.weight = None
        self.mu = None
        self.sigma = None
        self.K = None
        self.B = None

    def check(self, pixel, mu, sigma):
        '''
        check whether a pixel match a Gaussian distribution. Matching means pixel is less than
        2.5 standard deviations away from a Gaussian distribution.
        '''
        x = np.mat(np.reshape(pixel, (3, 1)))
        u = np.mat(mu).T
        sigma = np.mat(sigma)
        # calculate Mahalanobis distance
        d = np.sqrt((x-u).T*sigma.I*(x-u))
        if d < 2.5:
            return True
        else:
            return False

    def train(self, K=4):
        '''
        train model
        '''
        self.K = K
        file_list = []
        # file numbers are from 1 to train_number
        for i in range(self.train_num):
            file_name = os.path.join(self.data_dir, 'b%05d' % i + '.bmp')
            file_list.append(file_name)

        img_init = cv.imread(file_list[0])
        img_shape = img_init.shape
        self.img_shape = img_shape
        self.weight = np.array([[init_weight for j in range(self.img_shape[1])] for i in range(self.img_shape[0])])
        self.mu = np.array([[[init_u for k in range(self.K)] for j in range(img_shape[1])]
                             for i in range(img_shape[0])])
        self.sigma = np.array([[[init_sigma for k in range(self.K)] for j in range(img_shape[1])]
                             for i in range(img_shape[0])])

        self.B = np.ones(self.img_shape[0:2], dtype=np.int)
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                for k in range(self.K):
                    self.mu[i][j][k] = np.array(img_init[i][j]).reshape(1,3)
        for i in range(self.K):
            print('u:{}'.format(self.mu[100][100][i]))
        # update process
        for file in file_list:
            print('training:{}'.format(file))
            img=cv.imread(file)
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    # Check whether match the existing K Gaussian distributions
                    match = -1
                    for k in range(K):
                        if self.check(img[i][j], self.mu[i][j][k], self.sigma[i][j][k]):
                            match = k
                            break
                    # a match found
                    if match != -1:
                        mu = self.mu[i][j][k]
                        sigma = self.sigma[i][j][k]
                        x = img[i][j].astype(np.float)
                        delta = x - mu
                        rho = self.alpha * mv_norm.pdf(img[i][j], mu, sigma)
                        self.weight[i][j] = (1 - self.alpha) * self.weight[i][j]
                        self.weight[i][j][match] += self.alpha
                        # self.weight[i][j][k] = self.weight[i][j][k] + self.alpha*(m - self.weight[i][j][k])
                        self.mu[i][j][k] = mu + rho * delta
                        self.sigma[i][j][k] = sigma + rho * (np.matmul(delta, delta.T) - sigma)
                    # if none of the K distributions match the current value
                    # the least probable distribution is replaced with a distribution
                    # with current value as its mean, an initially high variance and low rior weight
                    if match == -1:
                        w_list = [self.weight[i][j][k] for k in range(K)]
                        id = w_list.index(min(w_list))
                        # weight keep same, replace mean with current value and set high variance
                        self.mu[i][j][id] = np.array(img[i][j]).reshape(1, 3)
                        self.sigma[i][j][id] = np.array(init_sigma)
            print('img:{}'.format(img[100][100]))
            print('weight:{}'.format(self.weight[100][100]))
            self.reorder()
            for i in range(self.K):
                print('u:{}'.format(self.mu[100][100][i]))



    def reorder(self, T=0.90):
        '''
        reorder the estimated components based on the ratio pi / the norm of standard deviation.
        the first B components are chosen as background components
        the default threshold is 0.75
        '''
        for i in range(self.img_shape[0]):
            for j in range(self.img_shape[1]):
                k_weight = self.weight[i][j]
                k_norm = np.array([norm(np.sqrt(self.sigma[i][j][k])) for k in range(self.K)])
                ratio = k_weight/k_norm
                descending_order = np.argsort(-ratio)

                self.weight[i][j] = self.weight[i][j][descending_order]
                self.mu[i][j] = self.mu[i][j][descending_order]
                self.sigma[i][j] = self.sigma[i][j][descending_order]

                cum_weight = 0
                for index, order in enumerate(descending_order):
                    cum_weight += self.weight[i][j][index]
                    if cum_weight > T:
                        self.B[i][j] = index + 1
                        break
                # if self.B[i][j] == self.K:
                #     self.B[i][j] = self.K - 1


    def infer(self, img):
        '''
        infer whether its background or foregound
        if the pixel is background, both values of rgb will set to 255. Otherwise not change the value
        '''
        result = np.array(img)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(self.B[i][j]):
                    if self.check(img[i][j], self.mu[i][j][k], self.sigma[i][j][k]):
                        # [255, 255, 255] is white, the background color will be set to white
                        result[i][j] = [255, 255, 255]
                        break
        return result
