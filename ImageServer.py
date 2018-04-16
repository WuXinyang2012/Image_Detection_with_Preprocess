import os, time
from mvnc import mvncapi as mvnc
import numpy
import cv2
import os, sys
import KW_service as KW
#import tensorflow as tf

write_path = "/tmp/pipe.out"

if os.path.exists(write_path):
    os.remove(write_path)

os.mkfifo(write_path)

wf = os.open(write_path, os.O_CREAT | os.O_SYNC  | os.O_RDWR)

# Load graph
path_to_networks = './Retrained_model/'
graph_filename = 'ncs_V3.pb'
with open(path_to_networks + graph_filename, mode='rb') as f:
    graphfile = f.read()

# Load categories
categories = []
with open(path_to_networks + 'output_labels.txt', 'r') as f:
    for line in f:
        cat = line.split('\n')[0]
        if cat != 'classes':
            categories.append(cat)
    f.close()
    #print('Number of categories:', len(categories))
reqsize = 299

#load Devices
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print('No devices found')
    quit()
device = mvnc.Device(devices[0])
#device.CloseDevice()
device.OpenDevice()
graph = device.AllocateGraph(graphfile)


print('Server starts now.')

cap = cv2.VideoCapture(0)
def capture():
    ret, frame = cap.read()
    return frame

def ImageRead():
    return cv2.imread("./test1.jpg")

def inference(subimages, number):
    for k in range(number):
        print("Subimage %s" % k)
        image = subimages
        img = image
        #cv2.imshow("test",img)
        img = cv2.resize(img, (reqsize, reqsize))
        #cv2.imshow('%s' % k, img)
        print('Start download to NCS...')
        graph.LoadTensor(img.astype(numpy.float16), 'DecodeJpeg/contents:0')
        output, userobj = graph.GetResult()

        top_inds = output.argsort()[::-1][:5]

        print(''.join(['*' for i in range(79)]))
        print('inception-v3 on NCS')
        print(''.join(['*' for i in range(79)]))
        for i in range(5):
            print(top_inds[i], categories[top_inds[i]], output[top_inds[i]])

        print(''.join(['*' for i in range(79)]))


while True:
    s = os.read(wf, 1024)
    print "received msg: %s" % s
    if len(s) == 0:
        time.sleep(1)
        continue
    elif "exit" in s:
        break
    elif "inference" in s:
        image = ImageRead()
        #subimages, number = KW.main(image)
        inference(image, 1)
        os.write(wf, 'sleep')

print('Server exits now.')
graph.DeallocateGraph()
device.CloseDevice()
os.close(wf)
cap.release()
