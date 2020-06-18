
import grpc
from data_comm_pb2 import *
import data_comm_pb2_grpc
import data_comm_resources

with grpc.insecure_channel('localhost:50051') as channel:
    stub = route_guide_pb2_grpc.RouteGuideStub(channel)
    # feature = stub.GetFeature(point)
    response = stub.EstablishConnection(ConnectionParams(is_recording=False))
    action_dim = response.action_dim
    obs_dim = response.obs_dim
    vis_obs_dim = response.vis_obs_dim
    neos_do_recording = response.neos_do_recording
    agent_index = response.agent_index

    while True:
        obs = stub.GetObs(Empty())
        reward = obs.reward
        should_reset = obs.should_reset
        #ignoring visual obs for now
        
