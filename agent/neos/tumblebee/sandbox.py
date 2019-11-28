# import socket
# UDP_IP = "127.0.0.1"
# UDP_PORT = 9999
# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
#
# sock.bind((UDP_IP, UDP_PORT))
# # while True:
# #     data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
# #     print "received message:", data
#
#
# # data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
#
# # data = bytearray(b" " * 2048)
# # size = sock.recv_into(data)
#
# received = sock.recv(1)
#
# print(received)

import tensorflow as tf
import tensorflow.keras as keras
# model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=resized_image.shape, alpha=1.0, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
# model = keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
model = keras.applications.vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)



#%%

import json
class_idx = json.load(open("imagenet_class_index.json","rb"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
from imagenet_index_to_class import index_to_class
#%%


#%%
import cv2
import numpy as np; import matplotlib.pyplot as plt
cap = None
#%%
videourl = "udp://@:9999/"
if cap:
    cap.release()
cap = cv2.VideoCapture(videourl,cv2.CAP_FFMPEG)
cap

#%%
# ret, frame = cap.read()
# ret
# frame = frame[:,:,[2,1,0]]
# frame[:1,:1,:]

# plt.imshow(frame)
#%%
i=0
while cap.isOpened():
    ret, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

    i+=1
    if i%30==0:
        height, width, depth = frame.shape
        new_height = 224
        # new_height = 299
        new_width = int(width*(new_height/height))
        resized_image = cv2.resize(frame, dsize=(new_width, new_height))
        resized_image = resized_image[:,int(new_width/2-new_height/2):int(new_width/2-new_height/2)+new_height]

        pred = model.predict(np.expand_dims(resized_image,0))
        index = np.argmax(pred)
        # print(index_to_class[index])
        file = open("classification.txt","w")
        file.write(index_to_class[index])
        file.close()



perform_action("movZ")

import time
#%%
def perform_action(action):
    time_delay=0.05
    f = open("classification.txt","w")
    f.write(""); f.close()
    time.sleep(time_delay)
    f = open("classification.txt","w")
    f.write(action); f.close()
#%%

# #%%
#
# height, width, depth = frame.shape
# new_height = 224
# # new_height = 299
# new_width = int(width*(new_height/height))
# resized_image = cv2.resize(frame, dsize=(new_width, new_height))
#
# resized_image = resized_image[:,int(new_width/2-new_height/2):int(new_width/2-new_height/2)+new_height]
#
# plt.imshow(resized_image)
# #%%
#
# pred = model.predict(np.expand_dims(resized_image,0))
#
# pred.shape
# pred[0,index]
# pred[0,282]
# index = np.argmax(pred)
# pred.shape
#
# class_idx[str(index)]
# index_to_class[index]
#
# for idx,prob in sorted(enumerate(pred[0]),key=lambda x: -x[1])[:10]:
#     print(idx2label[int(idx)], prob)
