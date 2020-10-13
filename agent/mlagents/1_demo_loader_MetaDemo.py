import io
import os
from unittest import mock
import numpy as np
import pytest
import tempfile

from mlagents_envs.communicator_objects.demonstration_meta_pb2 import (
    DemonstrationMetaProto,
)
from mlagents.trainers.tests.mock_brain import (
    create_mock_3dball_behavior_specs,
    setup_test_behavior_specs,
)
from mlagents.trainers.demo_loader import (
    load_demonstration,
    demo_to_buffer,
    get_demo_files,
    write_delimited,
)

#%%

print("#########################################")
# demo_file=sys.argv[1]
demo_file="..\\..\\environment\\neos\\UnityMiddleWare2\\Assets\\Demonstrations\\NeosAgent_4.demo"
# demo_file="..\\..\\environment\\neos\\built_env\\Unity Environment_Data\\Demonstrations\\betatest2_0.demo"
group_spec, info_action_pairs, total_expected = load_demonstration(demo_file)
# group_spec, info_action_pairs, total_expected = load_demonstration("..\\..\\environment\\neos\\built_env\\Unity Environment_Data\\Demonstrations\\betatest2.demo")
# group_spec, info_action_pairs, total_expected = load_demonstration("D:\code\\temp\\built_env\\Unity Environment_Data\\Demonstrations\\betatest2_4.demo")
# group_spec, info_action_pairs, total_expected = load_demonstration("D:\code\\temp\\built_env\\Unity Environment_Data\\Demonstrations\\older\\betatest2_3.demo")
# group_spec, info_action_pairs, total_expected = load_demonstration("D:\code\\temp\\built_env\\Unity Environment_Data\\Demonstrations\\older\\betatest2_0.demo")
print(group_spec)
len(info_action_pairs)
type(info_action_pairs)
get_obs = lambda pair: np.array(pair.agent_info.observations[0].float_data.data)
get_actions = lambda pair: np.array(pair.action_info.vector_actions)

obs = np.stack(list(map(get_obs,info_action_pairs)))
acts = np.stack(list(map(get_actions,info_action_pairs)))

obs.shape
acts.shape

np.save("circling_box_obs",obs)
np.save("circling_box_acts",acts)

group_spec
print("#######")
print("Example obs-action pair")
print(info_action_pairs[0])
print("#######")
# type(info_action_pairs)
# len(info_action_pairs)
# info_action_pairs[0]

# type(info_action_pairs[0])

# list(info_action_pairs[0].agent_info.observations[1].float_data.data)[0])

float_obs=[list(pair.agent_info.observations[1].float_data.data) for pair in info_action_pairs]
import json
open("..\\..\\environment\\neos\\built_env\\Unity Environment_Data\\Demonstrations\\current_demo_floats.json","w").write(json.dumps(float_obs))
open("..\\..\\environment\\neos\\UnityMiddleWare2\\Assets\\Demonstrations\\current_demo_floats.json","w").write(json.dumps(float_obs))
# json.load(open("test.json","r"))
# info_action_pairs[0].agent_info.observations[0].compressed_data

# info_action_pairs[100]

print("#######")
print( total_expected)
print("#######")


# print("#########################################")
#
# group_spec, info_action_pairs, total_expected = load_demonstration("..\..\environment\neos\built_env\Unity Environment_Data\Demonstrations\betatest2.demo")
# print(group_spec)
# print("#######")
# print( info_action_pairs)
# print("#######")
# print( total_expected)
# print("#######")
