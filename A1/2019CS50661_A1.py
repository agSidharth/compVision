import numpy as np
import os
import cv2 as cv
import sys

# SID:: ??
init_weight = [0.7, 0.11, 0.1, 0.09]
init_sigma_factor = 0.01
# SID:: ??

class GMM():

	def toGray(self,img):
		return cv.cvtColor(img,cv.COLOR_BGR2GRAY)

	def __init__(self,data_dir,alpha,N = 10,K = 4):

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

		self.weights = []
		self.mu = []
		self.sigma = []
		self.history = []

		for i in range(self.img_shape[0]):
			tempW = []
			tempM = []
			tempS = []
			tempH = []

			for j in range(self.img_shape[1]):
				tempW.append(init_weight)
				tempM.append([self.first_img[i][j]]*self.K)
				tempS.append([init_sigma_factor]*self.K)
				tempH.append([self.first_img[i][j]]*self.N)

			self.weights.append(tempW)
			self.mu.append(tempM)
			self.sigma.append(tempS)
			self.history.append(tempH)

		self.weights = np.array(self.weights)
		self.mu = np.array(self.mu)
		self.sigma = np.array(self.sigma)
		self.history = np.array(self.history)
		self.B = np.ones(self.img_shape)

	def trainImg(self,img):
		for idx in range(self.img_shape[0]):
			for jdx in range(self.img_shape[1]):
				match = False

				temp_history = self.history[idx][jdx][1:]
				#print(len(self.history[idx][jdx]))
				self.history[idx][jdx] = np.append(temp_history,[img[idx][jdx]])

				for kdx in range(self.K):

					if (abs(img[idx][jdx] - self.mu[idx][jdx][kdx])/self.sigma[idx][jdx][kdx] < 2.5):
						match = True
						self.weights[idx][jdx][kdx] = (1-self.alpha)*self.weights[idx][jdx][kdx] + self.alpha
						# SID:: ??
						self.mu[idx][jdx][kdx] = np.mean(self.history[idx][jdx])
						self.sigma[idx][jdx][kdx] = np.std(self.history[idx][jdx])
						# SID:: ??

						if self.sigma[idx][jdx][kdx]==0:
							self.sigma[idx][jdx][kdx] = init_sigma_factor

				if not match:
					leastW = self.weights[idx][jdx][0]
					leastW_index = 0

					for kdx in range(self.K):
						if leastW>self.weights[idx][jdx][kdx]:
							leastW = self.weights[idx][jdx][kdx]
							leastW_index = kdx

					self.mu[idx][jdx][leastW_index] = img[idx][jdx]
					self.sigma[idx][jdx][leastW_index] = init_sigma_factor


	def train(self):
		for file in self.file_list:
			img = self.toGray(cv.imread(file))
			print(file)
			self.trainImg(img)
			self.reorder()

	def reorder(self,T = 0.90):
		for idx in range(self.img_shape[0]):
			for jdx in range(self.img_shape[1]):

				ratio = -1*self.weights[idx][jdx]/100*self.sigma[idx][jdx]
				descending_order = np.argsort(ratio)

				self.weights[idx][jdx] = self.weights[idx][jdx][descending_order]/np.sum(self.weights[idx][jdx])

				#print(self.weights[idx][jdx])
				#print(np.sum(self.weights[idx][jdx]))

				self.mu[idx][jdx] = self.mu[idx][jdx][descending_order]
				self.sigma[idx][jdx] = self.sigma[idx][jdx][descending_order]

				weightsSum = 0
				for kdx in range(self.K):
					weightsSum += self.weights[idx][jdx][kdx]
					if weightsSum>T:
						self.B[idx][jdx] = kdx
						break


	def infer(self,img):
		img = self.toGray(img)
		result = np.array(img)

		for idx in range(img.shape[0]):
			for jdx in range(img.shape[1]):
				for kdx in range(int(self.B[idx][jdx])+1):

					if (abs(img[idx][jdx] - self.mu[idx][jdx][kdx])/self.sigma[idx][jdx][kdx] < 2.5):
						result[idx][jdx] = 255
						break

		return result


if __name__=="__main__":
	data_dir = os.path.join(sys.argv[1],"input")
	gmm = GMM(data_dir,0.10)

	print("1:Training started")
	#gmm.train()
	print("2:Training finished")

	index = 0
	out_dir = os.path.join(data_dir,"2019CS50661_output")
	inputList = os.listdir(data_dir)

	for file in inputList:
		img = cv.imread(os.path.join(data_dir,file))
		newImg = gmm.infer(img)
		cv.imwrite(os.path.join(out_dir,file+".png"),newImg)
		index += 1


	







