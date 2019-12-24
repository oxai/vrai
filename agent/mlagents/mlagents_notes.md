
What were previously different "brains" now are different "behaviours"

_Config file_
Default section applies to all behaviours
then can have sections that apply to particular behaviours

Here I put bits of the code that are most relevant to modify stuff in mlagents
All folders are relative to root folder of mlagents library

## Main entry point

`mlagents-learn` calls this

`trainers/learn.py`
```
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import SideChannel
from mlagents.trainers.trainer_controller import TrainerController
from mlagents.trainers.trainer_util import load_config, TrainerFactory
from mlagents.trainers.sampler_class import SamplerManager

def run_training(
    #STUFF THAT DEALS WITH UNITY ENVIRONMENT
    #this sets up the UnityEnvironment with several options
    env_factory = create_environment_factory(
    engine_config = EngineConfig(options...
    env_manager = SubprocessEnvManager(env_factory, engine_config, options.num_envs)
    maybe_meta_curriculum = try_create_meta_curriculum(
    #STUFF THAT DEALS WITH TRAINING
    #uses trianer_config
    # uses SamplerManager
    sampler_manager, resampling_interval = create_sampler_manager(
    trainer_factory = TrainerFactory(
        trainer_config,
        ...
        maybe_meta_curriculum,
    )
    # Create controller and begin training.
    tc = TrainerController(
        trainer_factory,
        maybe_meta_curriculum,
        ...
        sampler_manager,
    )
    # Begin training
    tc.start_learning(env_manager)
```


### `trainers/trainer_util.py`

```

from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.sac.trainer import SACTrainer


class TrainerFactory:
    def generate(self, brain_parameters: BrainParameters) -> Trainer:
        return initialize_trainer(
        ...


def initialize_trainer(
    trainer_config: Any,
    brain_parameters: BrainParameters,
    summaries_dir: str,
    run_id: str,
    model_path: str,
    keep_checkpoints: int,
    train_model: bool,
    load_model: bool,
    seed: int,
    meta_curriculum: MetaCurriculum = None,
    multi_gpu: bool = False,
) -> Trainer:
   Initializes a trainer given a provided trainer configuration and brain parameters, as well as
    some general training session options.

    :param trainer_config: Original trainer configuration loaded from YAML
    :param brain_parameters: BrainParameters provided by the Unity environment
    :param summaries_dir: Directory to store trainer summary statistics
    :param run_id: Run ID to associate with this training run
    :param model_path: Path to save the model
    :param keep_checkpoints: How many model checkpoints to keep
    :param train_model: Whether to train the model (vs. run inference)
    :param load_model: Whether to load the model or randomly initialize
    :param seed: The random seed to use
    :param meta_curriculum: Optional meta_curriculum, used to determine a reward buffer length for PPOTrainer
    :param multi_gpu: Whether to use multi-GPU training

...

    if trainer_type == "offline_bc":
        raise UnityTrainerException(
            "The offline_bc trainer has been removed. To train with demonstrations, "
            "please use a PPO or SAC trainer with the GAIL Reward Signal and/or the "
            "Behavioral Cloning feature enabled."
        )
    elif trainer_type == "ppo":
        trainer = PPOTrainer(
            brain_parameters,
            min_lesson_length,
            trainer_parameters,
            train_model,
            load_model,
            seed,
            run_id,
            multi_gpu,
        )
    elif trainer_type == "sac":
        trainer = SACTrainer(
            brain_parameters,
            min_lesson_length,
            trainer_parameters,
            train_model,
            load_model,
            seed,
            run_id,
        )
'''
brain_parameters are set when calling the `generate` method of the TrainerFactory
which I think is called within TrainerController
'''

```

## Main training loop

### `trainers/trainer_controller.py`

```

class TrainerController(object):
    def __init__(
        self,
        trainer_factory: TrainerFactory,
        model_path: str,
        summaries_dir: str,
        run_id: str,
        save_freq: int,
        meta_curriculum: Optional[MetaCurriculum],
        train: bool,
        training_seed: int,
        sampler_manager: SamplerManager,
        resampling_interval: Optional[int],
    ):
        """
        :param model_path: Path to save the model.
        :param summaries_dir: Folder to save training summaries.
        :param run_id: The sub-directory name for model and summary statistics
        :param save_freq: Frequency at which to save model
        :param meta_curriculum: MetaCurriculum object which stores information about all curricula.
        :param train: Whether to train model, or only run inference.
        :param training_seed: Seed to use for Numpy and Tensorflow random number generation.
        :param sampler_manager: SamplerManager object handles samplers for resampling the reset parameters.
        :param resampling_interval: Specifies number of simulation steps after which reset parameters are resampled.
        """
        ...
    def _write_training_metrics(self):
        for brain_name in self.trainers.keys():
            if brain_name in self.trainer_metrics:
                self.trainers[brain_name].write_training_metrics()
    def _reset_env(self, env: EnvManager) -> List[EnvironmentStep]:
        """Resets the environment.

        Returns:
            A Data structure corresponding to the initial reset state of the
            environment.
        """
    def write_to_tensorboard(self, global_step: int) -> None:
    def start_trainer(self, trainer: Trainer, env_manager: EnvManager) -> None:

    ######
    # This is the main function, that begins the training loop
    def start_learning(self, env_manager: EnvManager) -> None:
    #it resets environment
    # it generates the trainers as follows
                        trainer = self.trainer_factory.generate(
                            env_manager.external_brains[name]
                        )
                        self.start_trainer(trainer, env_manager)

    # as we can see the `brain_parameters` needed to instantiate the trainer are just the variables defined in the Behaviour component in the agent in the Unity environemnt
    # while training not done
    # look for brain names defined in unity environment
    # btw, it seems to allow for the possibility of new brains being created while training is going on
    # inside the training loop it runs the environment as
                n_steps = self.advance(env_manager)
                for i in range(n_steps):
                    global_step += 1
                    self.reset_env_if_ready(env_manager, global_step)
    # ^ reseting the environment is used for curriculum learning, generalization learning, where the environent can change with different parameters, according to lesson
    # note that _reset_env is not the same as starting a new episode. That is handled by the env (EnvManager) class itself, just like an openai gym env would do.
    # we then save model, checkpoints, metrics, write to tensorboard etc

    # here is the function inside the main training loop
    def advance(self, env: EnvManager) -> int:
        new_step_infos = env.step()
        #WE FIRST GET THE EXPERIENCES FROM ENVIRONMENT
        #and send them to trainer
        for step_info in new_step_infos:
            for brain_name, trainer in self.trainers.items():
                if step_info.has_actions_for_brain(brain_name):
                    trainer.add_experiences(
                        step_info.previous_all_brain_info[brain_name],
                        step_info.current_all_brain_info[brain_name],
                        step_info.brain_name_to_action_info[brain_name].outputs,
                    )
                    trainer.process_experiences(
                        step_info.previous_all_brain_info[brain_name],
                        step_info.current_all_brain_info[brain_name],
                    )
        #WE THEN TRAIN
        #and update the policy in the environment
        #not sure what that ^ does, 
        #is it runnnig the training agent on unity using barracuda while its training too?
        for brain_name, trainer in self.trainers.items():
            if self.train_model and trainer.get_step <= trainer.get_max_steps:
                trainer.increment_step(len(new_step_infos))
                if trainer.is_ready_update(): #we train when we have filled the buffer I think
                    # Perform gradient descent with experience buffer
                    trainer.update_policy()
                    env.set_policy(brain_name, trainer.policy)
            else:
                # Avoid memory leak during inference
                # Eventually this whole block will take place in advance()
                # But currently this only calls clear_update_buffer() in RLTrainer
                # and nothing in the base class
                trainer.advance()
        return len(new_step_infos)

```

_Summary of important methods of trainer, which are used in main training loop above_

`trainer.add_experiences` (in base class RLTrainer in `rl_trainer.py`
`trainer.process_experiences`
`trainer.update_policy`
`trainer.advance`
`trainer.is_ready_update`
`trainer.increment_step`

These contain the `brain_info`s which contain experiences in the last step I think, used by trainers
```
step_info.previous_all_brain_info[brain_name],
step_info.current_all_brain_info[brain_name],
```
they contain visual_observations, vector_observations

the 'take_action_outputs` have the outputs of the neural net, including actions, entropy, learning rate, value predictions


## Trainers

### `trainers/rl_trainer.py`

```

class RLTrainer(Trainer):
    """
    This class is the base class for trainers that use Reward Signals.
    Contains methods for adding BrainInfos to the Buffer.
    """
    # the buffers that store the recent experiences of all the agents (which are using this trainer)
    self.processing_buffer = ProcessingBuffer()
    self.update_buffer = AgentBuffer()
    def construct_curr_info(self, next_info: BrainInfo) -> BrainInfo:
        """
        Constructs a BrainInfo which contains the most recent previous experiences for all agents
        which correspond to the agents in a provided next_info.
        :BrainInfo next_info: A t+1 BrainInfo.
        :return: curr_info: Reconstructed BrainInfo to match agents of next_info.
        """
    def add_experiences(
        self,
        curr_info: BrainInfo,
        next_info: BrainInfo,
        take_action_outputs: ActionInfoOutputs,
    ) -> None:
        """
        Adds experiences to each agent's experience history.
        :param curr_info: current BrainInfo.
        :param next_info: next BrainInfo.
        :param take_action_outputs: The outputs of the Policy's get_action method.
        """

        #FIRST COMPUTE REWARDS
        # Evaluate and store the reward signals
        tmp_reward_signal_outs = {}
        for name, signal in self.policy.reward_signals.items():
            tmp_reward_signal_outs[name] = signal.evaluate(
                curr_to_use, take_action_outputs["action"], next_info
            )
        # Store the environment reward
        tmp_environment = np.array(next_info.rewards, dtype=np.float32)

        rewards_out = AllRewardsOutput(
            reward_signals=tmp_reward_signal_outs, environment=tmp_environment
        )


        #THEN WE ADD EXPERIENCES TO processing_buffer
        for agent_id in next_info.agents:
        ...
            '''
            it adds (prev_observation, next_observation, memory, done, action)
            action lives inside self.policy.retrive_previous_action
            instead of the brain info for some reason (why did they do it this way?)
            '''
            # we also add rewards
            # Add the value outputs if needed
            self.add_rewards_outputs(
                rewards_out, values, agent_id, idx, next_idx
            )


    def add_rewards_outputs(
        self,
        rewards_out: AllRewardsOutput,
        values: Dict[str, np.ndarray],
        agent_id: str,
        agent_idx: int,
        agent_next_idx: int,
    ) -> None:
        """
        Takes the value and evaluated rewards output of the last action and store it
        into the training buffer. We break this out from add_experiences since it is very
        highly dependent on the type of trainer.
        :param take_action_outputs: The outputs of the Policy's get_action method.
        :param rewards_dict: Dict of rewards after evaluation
        :param agent_id: the Agent we're adding to.
        :param agent_idx: the index of the Agent agent_id in the current brain info
        :param agent_next_idx: the index of the Agent agent_id in the next brain info
        """

    def add_policy_outputs(
        self, take_action_outputs: ActionInfoOutputs, agent_id: str, agent_idx: int
    ) -> None:
        """
        Takes the output of the last action and store it into the training buffer.
        We break this out from add_experiences since it is very highly dependent
        on the type of trainer.
        :param take_action_outputs: The outputs of the Policy's get_action method.
        :param agent_id: the Agent we're adding to.
        :param agent_idx: the index of the Agent agent_id
        """
        raise UnityTrainerException(
            "The add_policy_outputs method was not implemented."
        )



```

`processing_buffer[agent_id].last_brain_info` holds the last `brain_info`
`processing_buffer[agent_id].last_take_action_outputs` holds the last `take_action_outputs`

### `trainers/ppo/trainer.py`

```

class PPOTrainer(RLTrainer):
        """
        Responsible for collecting experiences and training PPO model.
        :param trainer_parameters: The parameters for the trainer (dictionary).
        :param reward_buff_cap: Max reward history to track in the reward buffer
        :param training: Whether the trainer is set for training.
        :param load: Whether the model should be loaded.
        :param seed: The seed the model will be initialized with
        :param run_id: The identifier of the current run
        """
    def process_experiences(
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param current_info: current BrainInfo.
        :param next_info: next BrainInfo.
        """
        '''
        Basically puts stuff from general processing_buffer
        to the more algorithm-specific update_buffer
        for ppo this involves computing the advantages
        '''
    def update_policy(self):
        """
        Uses demonstration_buffer to update the policy.
        The reward signal generators must be updated in this method at their own pace.
        """
```


_Questions_

What's the timing tree?
somthing that profiles the code, and checks how fast different parts of the code run
uses @timed decorator. Where is the info accessible?
ah ok, it's in `{summaries_dir}/{run_id}_timers.json`

What does the `sampler_manager` (used in TrainingController) do?
where is `add_policy_outputs` in `ppo/trainer.py` used? In `rl_trainer.py`

I think it doesn't support policies that don't have value_heads

what does `env.update_policy`? is it using barracuda to run policy on environment, even during training?

- How to do generalization training
- How to do imitation learning
 - how to do offline
