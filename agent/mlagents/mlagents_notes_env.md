
`trainers/learn.py`

```
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import BaseEnv
from mlagents.trainers.subprocess_env_manager import SubprocessEnvManager

    env_factory = create_environment_factory(
        options.env_path,
        options.docker_target_name,
        options.no_graphics,
        run_seed,
        port,
        options.env_args,
    )
    engine_config = EngineConfig(
        options.width,
        options.height,
        options.quality_level,
        options.time_scale,
        options.target_frame_rate,
    )
    env_manager = SubprocessEnvManager(env_factory, engine_config, options.num_envs)

```

then SubprocessEnvManager is the class that inherits fron EnvManager (in `trainers/env_manager.py`) and implements the important `external_brains` method which gets the `brain_parameters` (aka simply `brains`) which are passed to the trainers.

`trainers/subprocess_env_manager.py`

```
    @property
    def external_brains(self) -> Dict[str, BrainParameters]:
        self.env_workers[0].send("external_brains")
        return self.env_workers[0].recv().payload

```

in `self.env_workers[0]` we have `UnityEnvWorker` objects (which are feed a process created with multiprocressing library which is the `worker` function in that same file

this worker process has an infinite loop which receives commands (`cmd: EnvironmentCommand = parent_conn.recv()`), which are sent by a parent process created by `UnityEnvWorker`, these commands then call python functions defined inside the `worker` function. The communication between these processes is managed by a multiprocessin pipe `parent_conn, child_conn = Pipe()` inside `SubprocessEnvManager.create_worker`. [#curiosity: Multiprocessing pipes uses sockets (BSD socket interface, if installed on Unix systems) or Windows named pipes (if installed on Windows)]

Ok, so what function does the `"extrnal_brains"` command call? this one

```
    def external_brains():
        result = {}
        for brain_name in env.get_agent_groups():
            result[brain_name] = group_spec_to_brain_parameters(
                brain_name, env.get_agent_group_spec(brain_name)
            )
        return result
```

which gets the agent groups (behavior names)

`group_spec_to_brain_parameters` creates brain from "`group_spec`" which does this

```
def group_spec_to_brain_parameters(
    name: str, group_spec: AgentGroupSpec
) -> BrainParameters:
    vec_size = np.sum(
        [shape[0] for shape in group_spec.observation_shapes if len(shape) == 1]
    )
    vis_sizes = [shape for shape in group_spec.observation_shapes if len(shape) == 3]
    cam_res = [CameraResolution(s[0], s[1], s[2]) for s in vis_sizes]
    a_size: List[int] = []
    if group_spec.is_action_discrete():
        a_size += list(group_spec.discrete_action_branches)
        vector_action_space_type = 0
    else:
        a_size += [group_spec.action_size]
        vector_action_space_type = 1
    return BrainParameters(
        name, int(vec_size), cam_res, a_size, [], vector_action_space_type
    )
```

we'll look at the definition of `BrainParameters` soon but first let's look at something more upstream, where the agent specs are gotten from the environment

on the other hand the agent specs themselves are gotten by `get_agent_group_spec` defined in `env/environments.py` as

```
    def get_agent_group_spec(self, agent_group: AgentGroup) -> AgentGroupSpec:
        self._assert_group_exists(agent_group)
        return self._env_specs[agent_group]
```

ok `self._env_specs` is updated in `_update_group_specs` by doing

```
    def _update_group_specs(self, output: UnityOutputProto) -> None:
        init_output = output.rl_initialization_output
        for brain_param in init_output.brain_parameters:
            # Each BrainParameter in the rl_initialization_output should have at least one AgentInfo
            # Get that agent, because we need some of its observations.
            agent_infos = output.rl_output.agentInfos[brain_param.brain_name]
            if agent_infos.value:
                agent = agent_infos.value[0]
                new_spec = agent_group_spec_from_proto(brain_param, agent)
                self._env_specs[brain_param.brain_name] = new_spec
                logger.info(f"Connected new brain:\n{brain_param.brain_name}")
```

the `output` object fed to this function is called in the `step` and `reset` environment functions, and is obtained from `self.communicator` as `outputs = self.communicator.exchange(step_input)` which is initialized as `self.communicator = self.get_communicator(worker_id, base_port, timeout_wait)` which calls

```
    def get_communicator(worker_id, base_port, timeout_wait):
        return RpcCommunicator(worker_id, base_port, timeout_wait)
```

what is this `RpcCommunnicator`? It's defined in `from .rpc_communicator import RpcCommunicator`, so let's look at `envs/rpc_communicator.py`, the `exchange` function is

```
    def exchange(self, inputs: UnityInputProto) -> Optional[UnityOutputProto]:
        message = UnityMessageProto()
        message.header.status = 200
        message.unity_input.CopyFrom(inputs)
        self.unity_to_external.parent_conn.send(message)
        self.poll_for_timeout()
        output = self.unity_to_external.parent_conn.recv()
        if output.header.status != 200:
            return None
        return output.unity_output
```

hmm the input is a `UnityInputProto`. hmm let's look at what was the input when `exchange` was called. back to `env/environment.py`. It is `step_input = self._generate_step_input(self._env_actions)` which does

```
    def _generate_step_input(
        self, vector_action: Dict[str, np.ndarray]
    ) -> UnityInputProto:
        rl_in = UnityRLInputProto()
        for b in vector_action:
            n_agents = self._env_state[b].n_agents()
            if n_agents == 0:
                continue
            for i in range(n_agents):
                action = AgentActionProto(vector_actions=vector_action[b][i])
                rl_in.agent_actions[b].value.extend([action])
                rl_in.command = STEP
        rl_in.side_channel = bytes(self._generate_side_channel_data(self.side_channels))
        return self.wrap_unity_input(rl_in)
```

seems to have the actions and side channel wrapped in some way. Anyway, let's focus on the output returned by `exchange`, which should have the agent specs and thus brain parameters, inside a property `rl_initialization_output` (see `_update_group_specs` above)...

The output is generated by `output = self.unity_to_external.parent_conn.recv()` where `self.unity_to_external = UnityToExternalServicerImplementation()` which is 

```
class UnityToExternalServicerImplementation(UnityToExternalProtoServicer):
    def __init__(self):
        self.parent_conn, self.child_conn = Pipe()

    def Initialize(self, request, context):
        self.child_conn.send(request)
        return self.child_conn.recv()

    def Exchange(self, request, context):
        self.child_conn.send(request)
        return self.child_conn.recv()
```

which extends `UnityToExternalProtoServicer` defined in `mlagents.envs.communicator_objects.unity_to_external_pb2_grpc`, so to file `envs/communicator_objects/unity_to_external_pb2_grpc.py` we go. `UnityToExternalProtoServicer` is just an emtpy class, so doesn't add more, but we see here also the method `add_UnityToExternalProtoServicer_to_server`, which is used in `envs/rpc_communicator.py` to add `self.unity_to_external` to a RPC server (this file was apparently created by a "gRPC Python protocol compiler plugin". hmm how does that work?).

This looks like a dead end. We need to undersand what process is `self.unity_to_external.parent_conn`. It seems I can't know without knowing how grpc works...

ok it seems we are missing .proto files inside `communicator_objects`. but still not sure where the parent and child processes of `self.unity_to_external` are tied to unity environments or whatever

Ok, I'll look at BrainParameters class next, maybe that should give me what I need to know.
Or hecc, maybe I should just look at how side_channels work!







