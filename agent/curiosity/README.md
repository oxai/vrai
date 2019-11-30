
## Learning-Progress RNN (LPRNN)

See the design of the architecture in `lprnn.pdf`. That document follows the convention:
 * Rectangles represent vectors (or scalars if of dimension 1).
 * Lines connecting rectangles represent feedforward NNs (typically MLPs)

The architecture is implemented in `learning_progress_rnn.py`

The training procedure is outlined in that document, but is explained in more detail in `train_lprnn.py`, which is also the script to train the network. In short,
 * Train `goal_encoder` and `goal_decoder` to autoencode/represent goals in latent `h`
 * Train `rnn` to produce a new `h` (from previous `h`, learning_progress, and observations) which encodes a good goal (good in the sense of having high expected learning progress)
 * Train `action_decoder` to produce an action which which is expected to have high `goal_reward`, i.e. which is expected to lead to an observation closely matching the goal encoded in `h`.

To train simply call `python3 train_lprnn.py` (needs mujoco, gym setup). Inside file there are variables to set whether to render (slow for training), and whether to evaluate on the openai-gym-provided goals (to test whether it has explored successfully to solve a variety of tasks)
