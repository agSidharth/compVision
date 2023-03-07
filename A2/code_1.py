import cv2
import os
import numpy as np

def gaussuian_filter(kernel_size, sigma=1, muu=0):

	x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
	                   np.linspace(-1, 1, kernel_size))
	dst = np.sqrt(x**2+y**2)

	# lower normal part of gaussian
	normal = 1/np.sqrt(2 * np.pi * sigma**2)

	# Calculating Gaussian filter
	gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normal
	return gauss

class Panorama:

	def __init__(self, input_path, output_path=""):

		self.frames = []
		self.output_path = output_path
		filenames = []
		for fn in os.listdir(input_path):
			filename, num = fn.split(".")[0].split(" ")
			filenames.append((int(num), fn))
		filenames.sort(key=lambda x: x[0])
		
		for fn in filenames:
			frame = cv2.imread(input_path + fn[1])
			frame = cv2.resize(frame, (1600, 1000))
			self.frames.append(frame)

		print("Number of frames found are {}".format(len(self.frames)))
		print("Each frame size is {}".format(self.frames[0].shape))

		self.find_interest_point()
		self.get_affines()
		self.get_panorama_dimensions()
		self.generate_panorama()
		
	def get_interest_points(self, frame):

		h, w, _ = frame.shape 
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Smoothening the image
		gray_frame = cv2.blur(gray_frame, (3, 3))

		# Gradient calculation
		grad_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
		grad_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)

		# Harris matrix calculation
		I_x_squared = grad_x*grad_x
		I_y_squared = grad_y*grad_y
		I_xy = grad_x*grad_y

		k = 1
		k_ = 0.05
		gaussian_window = gaussuian_filter(2*k+1)

		R = np.zeros(gray_frame.shape)
		w_x = cv2.filter2D(src=I_x_squared, ddepth=-1, kernel=gaussian_window)
		w_y = cv2.filter2D(src=I_y_squared, ddepth=-1, kernel=gaussian_window)
		w_xy = cv2.filter2D(src=I_xy, ddepth=-1, kernel=gaussian_window)
		det = w_x*w_y - (w_xy*w_xy)
		trace = w_x + w_y
		R = det - k_*(trace*trace)

		thr = 0.2
		idxs = np.where(R > thr*R.max())
		indexes = list(zip(idxs[0], idxs[1]))

		filter_indexes = []
		for i in range(len(indexes)):
			if (indexes[i][0] >= 4 and indexes[i][0] <= h-4 and indexes[i][1] >= 4 and indexes[i][1] <= w-4):
				filter_indexes.append(indexes[i])

		print("Number of interest points found: ", len(indexes))
		return filter_indexes

	def find_interest_point(self):

		self.keypoints = []
		for frame in self.frames:
			keypoint = self.get_interest_points(frame)
			self.keypoints.append(keypoint)

	def isProximal(self, img1, img2, x1, y1, l):
		thr = 100
		min_diff = 1000
		x_min = l[0][0]
		y_min = l[0][1]
		for i in range(len(l)):
			x2, y2 = l[i]
			diff = img2[x2-2:x2+2,y2-2:y2+2,:]-img1[x1-2:x1+2,y1-2:y1+2,:]
			diff = diff**2
			sum_diff = np.sqrt(np.sum(diff))
			if sum_diff <= min_diff:
				min_diff = sum_diff
				x_min = x2 
				y_min = y2 
		return (x_min, y_min)

	def proximity_search(self, u, v):

		key_points_frame_1 = self.keypoints[u]
		key_points_frame_2 = self.keypoints[v]

		frame_1 = self.frames[u]
		frame_2 = self.frames[v]
		
		src_points = []
		dst_points = []

		for (x1, y1) in key_points_frame_1:
			thr = 10
			l = []
			for (x2, y2) in key_points_frame_2:
				dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
				if dist < thr:
					l.append([x2, y2])

			if len(l) != 0:
				x_min, y_min = self.isProximal(frame_1, frame_2, x1, y1, l)
				src_points.append([y1, x1, 1])
				dst_points.append([y_min, x_min, 1])
		src_points = np.array(src_points)
		dst_points = np.array(dst_points)

		return (src_points, dst_points)

	def get_homography(self, u, v):

		# Takes from frame u to frame v
		print("Finding homography between {} and {}".format(u, v))
		src, dst = self.proximity_search(u, v)
		M=np.linalg.lstsq(src,dst,rcond=None)[0] 
		M=np.transpose(M)
		M[2]=np.array([0,0,1])
		return M


	def get_affines(self):

		l = len(self.frames)
		self.affines = [None] * l
		H = np.eye(3)
		self.affines[0] = H

		for i in range(1, l):
			self.affines[i] = np.matmul(self.affines[i-1], self.get_homography(i, i-1))
		return 

	def generate_panorama(self):
		w_min,w_max,h_min,h_max = self.final_corners
		H = h_max-h_min+1
		W = w_max-w_min+1
		l = len(self.frames)
		final_image = np.zeros((H, W, 3)).astype(int)
		for i in range(l):
			frame = self.frames[i]
			warped_frame = cv2.warpAffine(self.frames[i], self.affines[i][0:2], (W, H))
			index = np.where(final_image == [0,0,0])
			final_image[index] = warped_frame[index]
		final_image = np.array(final_image).astype(int)
		final_image_ = cv2.convertScaleAbs(self.filter(final_image))
		print(final_image_.shape)
		cv2.imwrite(os.path.join(self.output_path,"merged_image" + ".png"),final_image_)

	def filter(self, frame):
		for i in range(1, frame.shape[0]-1):
			for j in range(1, frame.shape[1]-1):
				diff1 = frame[i,j-1] - frame[i, j]
				diff2 = frame[i, j+1] - frame[i, j]
				diff1 = np.sqrt(diff1[0]**2 + diff1[1]**2 + diff1[2]**2)
				diff2 = np.sqrt(diff2[0]**2 + diff2[1]**2 + diff2[2]**2)

				val = np.sqrt(frame[i][j][0]**2 + frame[i][j][1]**2+ frame[i][j][2]**2)
				if (diff1 > 30 and diff2 > 30 and val < 600):
					frame[i,j,0] = (frame[i,j-1,0] + frame[i,j+1,0])//2
					frame[i,j,1] = (frame[i,j-1,1] + frame[i,j+1,1])//2
					frame[i,j,2] = (frame[i,j-1,2] + frame[i,j+1,2])//2
		return frame

	def get_panorama_dimensions(self):
		Corners=[]
		for i in range(0,len(self.frames)):
			H, W, _ = self.frames[i].shape
			corners = np.array([[0,0,1],[W-1,0,1],[W-1,H-1,1],[0,H-1,1]]).T
			corners_new_position = np.rint(np.matmul(self.affines[i],corners).T)
			new_left = min(0,np.int32(np.amin(corners_new_position[:,:1])))
			new_right = max(W-1,np.int32(np.amax(corners_new_position[:,:1])))
			new_top = min(0,np.int32(np.amin(corners_new_position[:,1:])))
			new_bottom = max(H-1,np.int32(np.amax(corners_new_position[:,1:])))
			L = [new_left,new_right,new_top,new_bottom]
			Corners.append(L)

		Corners=np.array(Corners).astype(np.int32)
		self.final_corners=(np.amin(Corners[:,0]),np.amax(Corners[:,1]),np.amin(Corners[:,2]),np.amax(Corners[:,3]))
		return


if __name__ == "__main__":

	panorama = Panorama("./dataset/1/", './output/1/')