Hi

page: http://oxai.org/socialvr-ai

discord channel: https://discord.gg/HQ8Crcw

ai in social vr tasks google doc https://docs.google.com/document/d/1QVXs_M1yCFSS0QU0ZtPwBJPwRGJJOXqJmiJ_OoFwIyk/edit?usp=sharing

original project proposal google doc https://docs.google.com/document/d/1_GhhYuYZBoCrgRzgp7adigoduGQmtOnopPdiNXcxktg/

## Instructions for use

### Set up

- Install NeosVR
- Install ML-Agents (version 0.15.X)
- Copy the dlls in `environment/neos/dlls/Libraries` to the `Libraries` folder in the Neos installation folder (from now on refered to as "Neos folder"), often in `C:\Program Files (x86)\Steam\steamapps\common\NeosVR`.
- Copy the file `run_neos.bat` in `environment/neos/dlls` to Neos folder.
- If you want to use one of the existing environments (right now, only up to date environment is the betali environment I'm using with the 18dof agent (actions are head, 2 hands)), then you'd need to ask it for me. Just register on Neos, send me a friend request (I'm guillefix), and I'll send you the world.
- The current training uses imitation learning, and that requires a demo file. I am not pusshing those because they can be too large for Github. They are in `environment/neos/built_env/Unity Environment_Data/Demonstrations`. I uploaded the current demo [here](environment/neos/UnityMiddleWare2/Assets/Demonstrations), so copy that to `environment/neos/built_env/Unity Environment_Data/Demonstrations` to use it if you want. This is quite a complex task to learn, sorry don't have a simple demo rn for testing :P maybe Rian can provide one he's working on.

### Running

- Open Neos with plugins, by running `run_neos.bat` (now found in Neos folder).
- Open the world (which would be wherever you left it after opening it from the message I sent, unless you have moved it somewhere else). This is done by double clicking on the orb. If using Neos in desktop mode, refer to here for the basic controls: http://wiki.neosvr.com/subdom/wiki/index.php?title=Basic_Controls. If you have VR, then we can meet in Neos and I can teach you the basics (or just ask me etc)
- Open cmd terminal and navigate to `agent/mlagents`. You can run `train_agent.bat`, but I recommend actually just using the command `mlagents-learn trainer_config.yaml --env="..\..\environment\neos\built_env\Unity Environment" --run-id=whatever-run-id-you-want --train` and also opening (a copy of) `gail_config.yaml`. Then you can change the training settings (which are the ones under `NeosAgent`). See https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-ML-Agents.md, https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-PPO.md, and https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Reward-Signals.md for documentation on what the different options mean, and intuitions on what they do.
- You can stop the training by pressing ctrl+c on the cmd.
- You can open also another cmd and navigate to `agent/mlagents` and run `tensorboard --logdir summaries --port 6006` and go to `localhost:6006` to see the tensorboard with learning metrics.

## Instructions for development

### Set up

- Do all of the setup for running as above
- Install Visual Studio (VS) Community edition

### Modifying the TeachableNeos.dll (the thing that connects Neos with external world, i.e. Unity for now)

- Open `environment/neos/TeachableNeos/TeachableNeos/TeachableNeos.csproj` which should open with VS.
- Edit there. `ServeData.cs` is the file with the Neos plugin itself. `DataCommImpl.cs` has the RPC functions (see [here](https://grpc.io/docs/tutorials/basic/csharp/) to learn more about RPC, and the gRPC library)
- Press Build>Build solution or ctrl+shift+B to build the solution (which should update the dll in the Neos folder, *if installed in the default location*)
- If you edit `basic_comm.proto` (which defines the RPC messages), then you'll need to save, close VS, and run `environment/neos/make_for_win.bat` to recompile them.

### Modifying the UnityMiddleware

- Just open the scripts in `Assets/VoidEnvironment/Scripts` in the UnityMiddleWare2 project. The main functionality is in `TestAgent.cs` which request obsevations from TeachableNeos server, and sends it actions. `TextureSensor.cs` and `TextureSensorComponent.cs` define a custom ML-Agents sensor to implement visual observsations via an image array received from Neos, rather than a Unity camera. Just save after making any modification and that's it.
- Build the environment into the built_env folder.

### Modifying Neos environment.

- You can create or modify any Neos environment. Explaining how to do this is beyond the scope of this documentation.
- TeachableNeos.dll gives a new LogiX node (a compoment in the Neos visual scripting language), which allows you to feed action demonstrations, observations (including visual observations with Neos Cameras), and then receive actions from the ML-Agents, to control your agent in Neos. You'll realistically need VR to use it for now (except maybe for the super-simplest examples, but doing LogiX in desktop mode is not pleasant). If interested, ask me. Otherwise, you'll have to rely on environments others build.

### Recording demos

- Go to the Neos world
- Press Pulse on set_recording in the in-world control panel
- Equip alphali avatar
- Do stuff
- Run `mlagents-learn trainer_config.yaml --env="..\..\environment\neos\built_env\Unity Environment" --run-id=whatever-run-id-you-want`
- Press Pulse on unset_recording in the in-world control panel, to return to normal "training mode".

### Others

- If Unity fails to connect to server. Try running stop_server and start_server in the in-Neos-world control panel

-----

_More stuff comming.._

* Try simpler tasks for betali. 
* OpenAI Gym environment. TODO: investigate how to parse demonstrations
* New algorithms, either using Gym, or figuring out how to add new ones to ML-Agents. E.g. adversarial behavioural cloning would be nice.
* Optimization. Try avatar without IK. Ok to speed up for environments without too many dynamics things
* Try parallelizing
* Try running on cloud. Figuring out how easy to run several Neos instances. Maybe easiest is several VMs, but they need really low latency communication between the spoke VMs and the hub VM running the training! So all in same cluster ideally.
* Some more of making stuff nice.
  * Compiling Unity Environment. 
  * Making python script to start mlagents-learn or whichever training script we use from within Neos itself.

-----

Version history

- **Alphali**. Only three actions (x-z velociy and y-rotation). Succesfully learnt simple find-and-reach task with only visual observations.
- **Betali**. Full d.o.f. control of head and both hands (position+rotation). Observations are visual + current pose. Actions and pose are parametrized relative to itself in a way that doesn't rely on any global coordinates, to help generalization. Learn some simple motion, but still not showing satisfactory results.

----

Disclaimer: `.bat` files in this repo have been tested and free from coronavirus.
