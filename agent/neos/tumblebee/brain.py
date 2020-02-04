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
model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling="avg", classes=1000)

model


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


from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
x = model.layers[-1].output
x = Dense(12, activation='softmax', name='qvalues')(x)
model1 = Model(inputs=model.input,outputs=x)
model1.compile(optimizer='sgd', loss='mean_squared_error')

actions = [sign+type+dir for type in ["mov", "rot"] for dir in ["X", "Y", "Z"] for sign in ["", "-"]]
rotations = [sign+type+dir for type in ["rot"] for dir in ["X", "Y", "Z"] for sign in ["", "-"]]
important_rotations = [sign+type+dir for type in ["rot"] for dir in ["X", "Y"] for sign in ["", "-"]]
translations = [sign+type+dir for type in ["mov"] for dir in ["X", "Y", "Z"] for sign in ["", "-"]]

#%%

len(actions)

#%%
import time
def perform_action(action):
    time_delay=0.05
    f = open("classification.txt","w")
    f.write(""); f.close()
    time.sleep(time_delay)
    f = open("classification.txt","w")
    f.write(action); f.close()

# perform_action("movZ")
#%%
agent_type = "dqn"
# target.shape
target = np.ones((1,512))/np.sqrt(512)
previous_action = 0
previous_image = np.zeros((1,224,224,3))
discount_factor = 0.9
exploration_probability = 0.2
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
        current_image = np.expand_dims(resized_image,0)

        pred = model.predict(np.expand_dims(resized_image,0))
        # reward = 0
        # if open("take.txt", "r").read() == "yes":
        #     target = model.predict(current_image)
        # elif open("reward.txt", "r").read() != "0":
        #     reward = int(open("reward.txt", "r").read())
        #
        # if agent_type=="dqn":
        #     qvalues = model1.predict(current_image)[0]
        #     qvalues_target = reward + discount_factor * np.max(qvalues)
        #     class_weights = np.zeros(len(qvalues))
        #     class_weights[previous_action] = 1.0
        #     target_vec = np.tile(np.expand_dims(qvalues_target,0),(1,12))
        #     model1.train_on_batch(previous_image,target_vec,class_weight=class_weights)
        #     if np.random.rand()<exploration_probability:
        #         index = np.random.randint(len(actions))
        #         perform_action(actions[index])
        #     else:
        #         previous_action = index = np.argmax(qvalues)
        #         perform_action(actions[index])
        # elif agent_type=="tumblebee":
        #     current = model.predict(current_image)
        #     overlap = np.dot(target[0,:],current[0,:])/(np.linalg.norm(current)*np.linalg.norm(target))
        #     prob_movZ = np.exp((overlap-0.9)*7)/(1+np.exp((overlap-0.9)*7))
        index = np.argmax(pred)
        #     # print(index_to_class[index])
        #     if np.random.rand()<prob_movZ:
        #         perform_action("movZ")
        #         perform_action("movZ")
        #     else:
        #         index = np.random.randint(len(actions))
        #         perform_action(actions[index])
        #         index = np.random.randint(len(actions))
        #         perform_action(actions[index])
        #         index = np.random.randint(len(actions))
        #         perform_action(actions[index])
        #         index = np.random.randint(len(actions))
        #         perform_action(actions[index])
        #         index = np.random.randint(len(actions))
        #         perform_action(actions[index])
        #         index = np.random.randint(len(actions))
        #         perform_action(actions[index])
        #         index = np.random.randint(len(actions))
        #         perform_action(actions[index])
        #         index = np.random.randint(len(actions))
        #         perform_action(actions[index])
        #         # index = np.random.randint(len(important_rotations))
        #         # perform_action(important_rotations[index])

        file = open("classification.txt","w")
        file.write(index_to_class[index])
        file.close()
        previous_image = current_image


# 1
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
