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
		# SID :: video part skipped
		
		result[noX,noY] = 255

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
	
	def return_integral(self,img):
		rows,cols = img.shape 
		dp = np.zeros(img.shape)

		prev = 0
		for c in range(cols):
			dp[0,c] = img[0,c] + prev
			prev = dp[0,c]
		
		prev = 0
		for r in range(rows):
			dp[r,0] = img[r,0] + prev
			prev = dp[r,0]
		
		for r in range(1,rows):
			for c in range(1,cols):
				dp[r,c] = img[r,c] + dp[r-1,c] + dp[r,c-1] - dp[r-1,c-1]
		return dp
	
	def remove_noise(self,img,thresh,width,length,stride):
		dp = self.return_integral(img)
		result = np.copy(img)

		r = 0
		while(r+width<img.shape[0]):
			c = 0
			while(c+length<img.shape[1]):
				thisSum = dp[r+width,c+length]
		
				if(r>0): thisSum -= dp[r-1,c+length]
				if(c>0): thisSum -= dp[r+width,c-1]
				if(r>0 and c>0): thisSum += dp[r-1,c-1]
				
				if(thisSum<thresh):
					result[r:(r+width+1),c:(c+length+1)] = 0

				c += stride
			r += stride

		"""
		cv.imshow("img",img)
		cv.waitKey(0)
		cv.imshow("result",result)
		cv.waitKey(0)
		"""

		return result

	def train(self,output_dir,useFilter = False,filter = None):
		index = 0
		for file in self.file_list:
			img = self.toGray(cv.imread(file))
			newImg = self.trainImg(img)
			
			if(useFilter): 
				newImg = self.remove_noise(newImg,filter["thresh"],filter["width"],filter["length"],filter["stride"])

			cv.imwrite(os.path.join(output_dir,str(index)+".png"),newImg)
			index += 1


if __name__=="__main__":
	alpha = 0.00001
	ourN = 10
	ourK = 4
	init_weight = [1/ourK]*ourK
	init_sigma_factor = 10
	filter = {'thresh' : 255*10,'width': 20, 'length': 20, 'stride': 20}

	data_dir = sys.argv[1]
	output_dir = sys.argv[2]
	useFilter = (sys.argv[3]=="filter") if len(sys.argv)>3 else False

	gmm = GMM(data_dir,alpha,N = ourN,K = ourK)

	if(useFilter):
		print("Filter is being used to clean noise")

	print("1:Training started")
	gmm.train(output_dir,useFilter,filter)
	print("2:Training finished")
