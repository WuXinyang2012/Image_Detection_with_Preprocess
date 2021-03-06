import numpy as np
import cv2
from sklearn.cluster import KMeans
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage

def main(path):

    inputImage = cv2.imread(path)
    reduced_inputImage = cv2.resize(inputImage, (0,0), fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    #cv2.imshow('reduced',reduced_inputImage)
    shifted = cv2.Meanshift(reduced_inputImage, (9,9), 0)
    cv2.imshow('test',shifted)
    m, n = shifted.shape[:2]
    #shifted = inputImage.copy()
    #cv2.imshow("MS",shifted)


    data = []
    for i in range(m):
            for j in range(n):
                x,y,z = shifted[i,j]
                data.append([x/256.0,y/256.0,z/256.0])


    label = KMeans(n_clusters=2).fit_predict(data)
    label = label.reshape([m,n])

    new_pic = np.zeros(( n, m), dtype=np.uint8)
    for i in range(n):
        for j in range(m):
            new_pic[i][j]= 255 if label[j][i] else 0 #rotation

    #cv2.imshow("tmp",new_pic)
    #shifted_tmp = cv2.pyrMeanShiftFiltering(tmp, 30, 50)
    #cv2.imshow("MS_tmp", shifted_tmp)


    # cv2.imshow("Thresh", thresh)
    if new_pic[0, 0] == 255:
        for i in range(n):
            for j in range(m):
                new_pic[i, j] = 255 - new_pic[i, j]
    #cv2.imshow("tmp", new_pic)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))
    #closed = cv2.morphologyEx(shifted_tmp, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("Close",closed);
    opened = cv2.morphologyEx(new_pic, cv2.MORPH_OPEN, kernel)
    #cv2.imshow("Open", opened);


    D = ndimage.distance_transform_edt(opened)
    localMax = peak_local_max(D, indices=False, min_distance=20,
        labels=opened)

    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=opened)

    k = 0
    sub_images = []
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue

        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(new_pic.shape, dtype="uint8")
        mask[labels == label] = 255
        #cv2.imshow('mask',mask)

        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)

        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        if r < 20:
            continue
        #cv2.circle(tmp, (int(x), int(y)), int(r), (0, 255, 0), 2)

        #cv2.putText(tmp, "#{}".format(k+1), (int(x) - 10, int(y)),
        #	cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


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
        sub_image = cv2.resize(sub_image, dsize=(200,200))
        sub_images.append(sub_image)
        cv2.imshow('%s'%k,sub_image)
        k = k + 1

    print("[INFO] {} unique segments found".format(k))
    # show the output image
#	cv2.imshow("Output", tmp)
#	cv2.waitKey(0)

    return sub_images, k
