import numpy as np
import PIL.Image as image
import argparse
import cv2
import os, sys
from sklearn.cluster import KMeans
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage

def mkdir(path):
    # delete space 
    path=path.strip()
    # delete possible "\\" symbols
    path=path.rstrip("\\")
 
    isExists=os.path.exists(path)
    if not isExists:
        print(path+' successfully made.')
        os.makedirs(path)
        return True
    else:
        print(path+' already exists.')
        return False


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
path = args["image"]
# load the image and perform pyramid mean shift filtering
# to aid the thresholding step
#print(path)
inputImage = cv2.imread(path)
reduced_inputImage = cv2.resize(inputImage,(0,0),fx=0.5,fy=0.5)
shifted = cv2.pyrMeanShiftFiltering(reduced_inputImage, 20, 50)
#shifted = inputImage.copy()
cv2.imshow("MS",shifted)

m,n = shifted.shape[:2]
data = []
for i in range(m):
        for j in range(n):
            x,y,z = shifted[i,j]
            data.append([x/256.0,y/256.0,z/256.0])

imgData,row,col = [data, m, n]

label = KMeans(n_clusters=2).fit_predict(imgData)
label = label.reshape([row,col])
pic_new = image.new("L", (row, col))
for i in range(row):
    for j in range(col):
        pic_new.putpixel((i,j), int(256/(label[i][j]+1)))
pic_new.save("KMeans_"+path, "JPEG")


tmp = cv2.imread("KMeans_"+path)
mm,nn = tmp.shape[:2]
cv2.imshow("tmp",tmp)
#shifted_tmp = cv2.pyrMeanShiftFiltering(tmp, 30, 50)
#cv2.imshow("MS_tmp", shifted_tmp)


"""
s = image.shape[:2]
m = shifted.copy()
for x in range(s[0]):
	for y in range(s[1]):
		r,g,b = m[x,y]
		if ((r / (g+b)) > 1.2):
			continue
		elif ((g / (r+b)) > 1.2):
			continue
		elif ((r+g)>1.2*b and abs(r-g) < 130 ):
			continue
		else:
				m[x,y]=0,0,0

cv2.imshow("RGB",m)
"""



# Otsu's thresholding
gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
print(thresh[0,0])
if thresh[0,0] == 255:
	for i in range(mm):
		for j in range(nn):
			thresh[i,j] = 255 - thresh[i,j]
cv2.imshow("Thresh",thresh)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))  
#closed = cv2.morphologyEx(shifted_tmp, cv2.MORPH_CLOSE, kernel)  
#cv2.imshow("Close",closed);  
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  
cv2.imshow("Open", opened);  


D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=20,
	labels=thresh)

markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)

subimage_path = path+"_subdir"
mkdir(subimage_path)

k = 0;
for label in np.unique(labels):
	# if the label is zero, we are examining the 'background'
	# so simply ignore it
	if label == 0:
		continue

	# otherwise, allocate memory for the label region and draw
	# it on the mask
	mask = np.zeros(gray.shape, dtype="uint8")
	mask[labels == label] = 255

	# detect contours in the mask and grab the largest one
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	c = max(cnts, key=cv2.contourArea)

	# draw a circle enclosing the object
	((x, y), r) = cv2.minEnclosingCircle(c)
	if r < 40:
		continue
	cv2.circle(tmp, (int(x), int(y)), int(r), (0, 255, 0), 2)

	cv2.putText(tmp, "#{}".format(k+1), (int(x) - 10, int(y)),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	k = k + 1;
	

	y1 = int(x) - int(r)
	y2 = int(x) + int(r)
	x1 = int(y) - int(r)
	x2 = int(y) + int(r)
	index = [x1,x2,y1,y2]
	for i in range(4):
		if index[i] < 0:
			index[i] = 0
	print(index)
	sub_image = reduced_inputImage[index[2]:index[3],index[0]:index[1]]
	img_path = './'+subimage_path+'/sub_image_'+str(k)+'.jpeg'
	cv2.imwrite( img_path, sub_image)
	cv2.imshow("sub",sub_image)

	
print("[INFO] {} unique segments found".format(k))
# show the output image
cv2.imshow("Output", tmp)
cv2.waitKey(0)
cv2.imwrite("output_"+path,tmp)
