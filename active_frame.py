import cv2
import os
import glob
import numpy as np
from operator import itemgetter
# import matplotlib.pyplot as plt
import math
import scipy.stats as stats

def main():
	video_dir = './KTH' #./testdata
	result_dir = './KTH-FSAF' #test-image
	loaddata(video_dir = video_dir, depth = 24, dest_forder=result_dir)

def save_image_to_file(frame_array, folder):
	for i in range(np.size(frame_array,axis=0)):
		cv2.imwrite(folder +"/" + format(i,'05d')+'.jpg', frame_array[i])

def loaddata(video_dir, depth, dest_forder):
	#video_dir can contain sub_directory
	dirs = os.listdir(video_dir)
	class_number = -1 
	#pbar = tqdm(total=len(files))
	for dir in dirs:
		path = os.path.join(video_dir, dir, '*.avi')
		files = sorted(glob.glob(path),key=lambda name: path )
		
		for filename in files:
			print('Extracting file:',filename)

			# frame_array = video3d_overlap(filename, depth)
			# frame_array = video3d_selected_active_frame(filename, depth)
			frame_array = full_selected_active_frame(filename, depth)
			
			newdir = dir + "/" + os.path.splitext(os.path.basename(filename))[0]
			directory = os.path.join(dest_forder,newdir)
			if not os.path.exists(directory):
				os.makedirs(directory)

			save_image_to_file(frame_array, directory)

def active_frames(frame_array):
	d=[] #euclid distance
	frames =[]
	for i in range(np.size(frame_array,axis=0)-1):
		d.append((np.linalg.norm(frame_array[i+1]-frame_array[i]),i,0))
	#Sort d[i] accending under first column of di

	d.sort(key=itemgetter(0)) #get the order of active frame
	d = normal_distribution(d) #assign each d one value based on normal distribution
	d.sort(key=itemgetter(1)) #re_order
	
	frames.append(frame_array[0])
	for i in range(1,np.size(d,axis=0)):
		temp_frame = frame_array[i] * d[i][2]
		frames.append(temp_frame)
	temp_frame = np.sum(frames, axis = 0)
	temp_frame = cv2.normalize(temp_frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)		
	return np.array(temp_frame)

#This function select numbers of the most active frame in a segment
def selected_active_frame(frame_array):
	max_euclidean_distance = 0
	temp_frame = frame_array[0] #assign first frame
	for i in range(np.size(frame_array,axis=0)-1):
		euclidean_distant = np.linalg.norm(frame_array[i+1]-frame_array[i])
		if euclidean_distant > max_euclidean_distance:
			max_euclidean_distance = euclidean_distant
			temp_frame = frame_array[i+1]

	return np.array(temp_frame)
#this function get the most active frame

def full_selected_active_frame(filename, depth):
	cap_images = read_video_from_file(filename)
	framearray = []
	distance = []
	
	for i in range(np.size(cap_images,axis=0)-1):
		distance.append((np.linalg.norm(cap_images[i+1]-cap_images[i]),i+1))
	frames = [item[1] for item in sorted(distance,key = itemgetter(0))[-depth:]]
	frames.sort()
	
	for i in range(np.size(frames,axis=0)):
		framearray.append(cap_images[frames[i]])
		

	return framearray

def video3d_selected_active_frame(filename, depth):
	cap_images = read_video_from_file(filename)
	framearray = []
	flatten_framearray = []
	nframe = np.size(cap_images,axis = 0)
	frames = [np.int(x * nframe / depth) for x in range(depth)]

	for i in range(np.size(frames,axis=0)):
		if i < np.size(frames,axis=0)-1:
			flatten_framearray = cap_images[frames[i]:frames[i+1]] 
		else: #last frame
			flatten_framearray = cap_images[frames[i]:nframe]
		newframe = selected_active_frame(flatten_framearray)
		framearray.append(newframe)

	return np.array(framearray)

def video3d_overlap(filename, depth = 16, overlap = 5, ):
	cap_images = read_video_from_file(filename)
	frame_array = []
	flatten_framearray = []
	nframe = np.size(cap_images,axis = 0)
	frames = [np.int(x * nframe / depth) for x in range(depth)]
	fromframe = 0
	toframe = 0
	for i in range(np.size(frames)):
		fromframe = frames[i] - overlap
		toframe = frames[i] + overlap
		if fromframe < 0: fromframe = 0
		if toframe > nframe-1: toframe = nframe-1
		flatten_framearray = cap_images[fromframe:toframe]
		frame = active_frames(flatten_framearray)

		frame_array.append(frame)
	return np.array(frame_array)

def read_video_from_file(filename):
	video_cap = cv2.VideoCapture(filename)
	nframe = np.int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frameWidth = np.int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frameHeight = np.int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))	
	j = 0
	ret = True
	cap_images = np.empty((nframe, frameHeight, frameWidth, 3))
	while (j < nframe  and ret):
		ret, cap_images[j] = video_cap.read()
		if ret != True: 
			cap_images = cap_images[0:j-1]
			break
		else:		
			j += 1

	return cap_images

def normal_distribution(d):
	dmax = max(l[0] for l in d)
	dmin = min(l[0] for l in d)
	mean = (dmax - dmin)/2
	sd = (mean - dmin)/3
	for i in range(np.size(d,axis=0)):
		temp = list(d[i])
		if dmax == dmin: #2frame is definitely the same
			temp[2] = 1	
		else:
			# temp[2] = 5*i+1
			temp[2] = alpha(16,i)
			# temp[2] = normpdf(i,mean,sd)
			# temp[2] = stats.norm(mean,sd).pdf(i)

		d[i] = tuple(temp)
	return d

#https://en.wikipedia.org/wiki/Normal_distribution#Probability_density_function
def normpdf(x, mean, sd):
    var = float(sd)**2
    pi = 3.1415926
    denom = (2*pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

def alpha(T, t):
	return 2*(T-t+1)-(T+1)*(Harmonic_number(T)-Harmonic_number(t-1))
	
def Harmonic_number(n):
	if n==0: 
		return 0
	return sum(1.0/i for i in range(1,n+1))  

if __name__ == '__main__':
	main()