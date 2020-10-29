# Imporing required libraries
import sys 
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from sklearn.cluster import KMeans

# Function for task 1 which will take the name of image as input
def task1(name):
	# Loading the image
    image = cv2.imread(name)
    # Rescaling the image to VGA size without changing aspect ratio
    image = cv2.resize(image, (600, 480),interpolation = cv2.INTER_LINEAR)
    # Getting rgb image to display original image later
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Converting to grayscale to extract keypoints and descriptors
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    # extracting keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(image,None)
    # Drawing keypoints with a circle around the keypoint whose radius is proportional to the scale of the keypoint 
    img = cv2.drawKeypoints(gray,keypoints,image,color=(0,255,0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img, keypoints, rgb

# Function to draw cross “+” at the location of the key point 
def drawKeyPts(im,keyp,th):
    for curKey in keyp:
        x=np.int(curKey.pt[0])
        y=np.int(curKey.pt[1])
        size = np.int(curKey.size)
        cv2.drawMarker(im, (x, y), (255,0,0), markerSize = th,);  
    return im  

# Function to display output
def draw(imgKey, rgb):
    dpi = 80
    height, width, depth = imgKey.shape
    figsize = (width / float(dpi)*2), (height / float(dpi)*2)
    figure = plt.subplots(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.title("Original Image(rescaled)")
    plt.imshow(rgb)
    plt.subplot(1, 2, 2)
    plt.imshow(imgKey)
    plt.axis('off')
    plt.title("Image with highlighted keypoints")
    plt.subplots_adjust(wspace=0.025, hspace=0.05)
    plt.show()

# Function for task 2 which will take the loaded images as input
def task2(img_array):
	# Rescaling the images to VGA size without changing aspect ratio
    for i in range(len(img_array)):
        img_array[i] = cv2.resize(image_array[i], (600, 480),interpolation = cv2.INTER_LINEAR)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_arr = []
    descriptors_arr = [] 
    features = []
    title = []
    # Extracting the names of images 
    for i in range(1, len(sys.argv)):
        title.append((str(sys.argv[i])).split(".")[0])
    # Extracting keypoints and descriptors for each image
    for i in img_array:
        keypoints, descriptors = sift.detectAndCompute(i,None)
        keypoints_arr.append(keypoints)
        descriptors_arr.extend(descriptors)
        features.append(descriptors)
    total_keypoints = 0
    # Displaying output (a) with keypoints in each image and total keypoints
    for i in range(len(keypoints_arr)):
        total_keypoints += len(keypoints_arr[i])
        print("# of keypoints in image {} is {}".format(title[i], len(keypoints_arr[i])))
    print("Total # of keypoints are", total_keypoints)
    print("............................")
    print("\n")
    k_count = 0
    # Creating an array of values of K according to total keypoints
    k_arr = [math.floor(0.05*total_keypoints), math.floor(0.1*total_keypoints), math.floor(0.2*total_keypoints)]
    # Generating BOVW for each K
    for k in k_arr:
    	# Clustering the SIFT descriptors
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(descriptors_arr)
        # Generating histogram for each image
        histograms = []
        for i in img_array:
            histo = np.zeros(k)
            kp, des = sift.detectAndCompute(i, None)
            nkp = np.size(kp)
            for d in des:
                idx = kmeans.predict([d])
                histo[idx] += 1/nkp
            histograms.append(histo)
        for i in range(len(histograms)):
            histograms[i] = histograms[i].astype('float32') 
        # Calculating chi - squared distances between each normalized histograms to find dissimilarity
        distances = []
        for i in range(len(histograms)):
            dis = []
            for j in range(len(histograms)):
                dis.append(cv2.compareHist(histograms[i], histograms[j], cv2.HISTCMP_CHISQR))
            distances.append(dis)
        for i in range(len(histograms)):
            for j in range(len(histograms)):
                distances[i][j] = round(distances[i][j], 2)
        ny = np.array(distances)
        k_values = [5, 10, 20]
        # Printing dissimilarity matrices for each K
        print("K = {}%*(total number of keypoints) = {}".format(k_values[k_count], k))
        k_count += 1
        print("\n")
        print("Dissimilarity matrix (Upper triangular matrix)")
        print("\t", end = "")
        for i in title:
            print(i, end = "\t")
        print("\n")
        for i in range(len(histograms)):
            print(title[i], end = "\t")
            for j in range(len(histograms)):
                print(np.triu(ny,k=0)[i][j], end = "\t")
            print("\n")
        print("\n")


# This will run when the python script runs
if __name__ == "__main__": 
	n = len(sys.argv)
	# If only 1 image is provided, run task 1 else run task 2
	if(n==2):
		name = sys.argv[1]
		img, keypoints, rgb = task1(name)
		imWithCross = drawKeyPts(img,keypoints,7)
		print("# of keypoints in " + name + " is", len(keypoints))
		draw(imWithCross, rgb)
	else:
		img_array = []
		for i in range(1, len(sys.argv)):
		    img_array.append(cv2.imread(sys.argv[i]))
		task2(img_array)   
