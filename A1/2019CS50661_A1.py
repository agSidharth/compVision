import numpy as np
import os
import cv2 as cv
import sys

class GMM():

	def toGray(self,img):
		return cv.cvtColor(img,cv.COLOR_BGR2GRAY)

	def __init__(self,data_dir,alpha,N,K):

		self.data_dir = data_dir
		self.alpha = alpha
		self.N = N
		self.K = K

		temp_file_list = os.listdir(self.data_dir)
		temp_file_list.sort()
		self.file_list = []
		for file in temp_file_list:
			self.file_list.append(os.path.join(self.data_dir,file))

		self.first_img = self.toGray(cv.imread(self.file_list[0]))
		self.img_shape = self.first_img.shape
		rows,cols = self.img_shape

		self.weights = np.ones((rows,cols,self.K))*(1/self.K)
		self.mu = np.ones((rows,cols,self.K))
		self.sigma = np.ones((rows,cols,self.K))*init_sigma_factor
		self.B = np.ones(self.img_shape)

		for k in range(self.K):
			self.mu[:,:,k] = self.first_img[:,:]
	
	def normal_dis(self,x,mean,sigma):
		result = (1/(np.sqrt(2*np.pi)*sigma))*np.exp((-1/2)*np.square((x-mean)/sigma))
		return result

	def trainImg(self,img):
		
		result = np.zeros(img.shape)

		#gaussian has matched with these indexes and indG
		img = img[:,:,np.newaxis]
		indX,indY,indG = np.where(abs(img-self.mu)/self.sigma <= 2.5)

		# gaussian has not matched with these indexes.
		minDis = np.min(abs(img-self.mu)/self.sigma,axis = 2)
		noX,noY = np.where(minDis>2.5)
		minGauss = np.argmin(self.weights,axis = 2)

		# SID :: ?? sorting and summation skipped.
		result[noX,noY] = img[noX,noY,0]

		self.weights = (1-self.alpha)*self.weights

		rho = self.alpha*self.normal_dis(img[indX,indY,0],self.mu[indX,indY,indG],self.sigma[indX,indY,indG])
		self.weights[indX,indY,indG] += (self.alpha)*self.weights[indX,indY,indG]
		self.mu[indX,indY,indG] = (1-rho)*self.mu[indX,indY,indG] + rho*img[indX,indY,0]

		tempSigma = np.square(self.sigma[indX,indY,indG])
		tempTerm = np.square(img[indX,indY,0] - self.mu[indX,indY,indG])
		self.sigma[indX,indY,indG] = np.sqrt((1-rho)*tempSigma + rho*(tempTerm))

		self.mu[noX,noY,minGauss[noX,noY]] = img[noX,noY,0]
		self.sigma[noX,noY,minGauss[noX,noY]] = init_sigma_factor

		return result
	
	def train(self,output_dir):
		index = 0
		for file in self.file_list:
			img = self.toGray(cv.imread(file))
			newImg = self.trainImg(img)
			cv.imwrite(os.path.join(output_dir,str(index)+".png"),newImg)
			index += 1


if __name__=="__main__":
	alpha = 0.01
	ourN = 10
	ourK = 4
	init_weight = [1/ourK]*ourK
	init_sigma_factor = 5

	data_dir = os.path.join(sys.argv[1],"input")
	gmm = GMM(data_dir,alpha,N = ourN,K = ourK)
	output_dir = sys.argv[2]

	print("1:Training started")
	gmm.train(output_dir)
	print("2:Training finished")


	







