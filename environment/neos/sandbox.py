'''
ah good old sandbox.py :)
'''

#RPC server stuff

import grpc
from concurrent import futures
# python -m grpc_tools.protoc -I./protos --python_out=. --grpc_python_out=. ./protos/basic_comm.proto

import basic_comm_pb2
from basic_comm_pb2 import Classification
import basic_comm_pb2_grpc
# import basic_comm_resources

class DataCommServicer(basic_comm_pb2_grpc.DataCommServicer):
    """Provides methods that implement functionality of data comm server."""

    def __init__(self):
        # self.db = basic_comm_resources.read_route_guide_database()
        pass

    def SendFeatures(self, request, context):
        print(request.thing)
        response = Classification(k=int(request.thing+request.thing2))
        return response



def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    basic_comm_pb2_grpc.add_DataCommServicer_to_server(
        DataCommServicer(), server)
    server.add_insecure_port('[::]:50052')
    server.start()
    server.wait_for_termination()

serve()

#%%

import tensorflow as tf
from tensorflow import keras

hidden_dim = 128
layer1 = keras.layers.Dense(hidden_dim)
layer2 = keras.layers.Dense(output_dim)
output1 = layer1(inputs)

model = keras.models.Model(inputs = inputs, outputs = output)
model.compile("adam",loss="mse")
model.fit(train_x,train_y)

#%%
n_steps=10
input_dim = 3
output_dim = 1
inputs = keras.layers.Input(shape=(n_steps,input_dim))
hidden_state_dim = 128
layer1 = keras.layers.GRU(hidden_state_dim,return_sequences=True)
layer2 = keras.layers.Dense(output_dim)
output1 = layer1(inputs)
output = layer2(output1)
#%%

model = keras.models.Model(inputs = inputs, outputs = output)
model.compile("adam",loss="mse")
model.fit(train_x,train_y)
