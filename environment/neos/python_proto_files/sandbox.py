
import grpc
from basic_comm_pb2 import *
import basic_comm_pb2_grpc
# import basic_comm_resources

# with grpc.insecure_channel('localhost:50051') as channel:
channel=grpc.insecure_channel('localhost:50052')
stub = basic_comm_pb2_grpc.DataCommStub(channel)
# feature = stub.GetFeature(point)
response = stub.EstablishConnection(ConnectionParams(is_recording=False))
action_dim = response.action_dim
obs_dim = response.obs_dim
vis_obs_dim = response.vis_obs_dim
# neos_do_recording = response.neos_do_recording
agent_index = response.agent_index

try:
    while True:
        obs = stub.GetObs(Empty())
        reward = obs.reward
        should_reset = obs.should_reset
        #ignoring visual obs for now
        print(obs.obs)
        stub.SendAct(NeosAction(action=[0.0]))
except (KeyboardInterrupt, SystemExit):
    stub.StopConnection(Empty())
