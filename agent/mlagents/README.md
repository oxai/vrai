
I am using the latest_release branch of the UnitySDK (get with `git clone --branch latest_release git@github.com:Unity-Technologies/ml-agents.git`) with the latest version of Unity. But not sure if this is necessary.
What *is* necessary is to install Barracuda preview Unity plugin (see [here](https://github.com/Unity-Technologies/ml-agents/issues/3027))

The mlagents python package I'm using is version 0.12.0 (pip install mlagents), but with the version of `mlagents_envs` in master branch. The way I got it to work is to

`git clone git@github.com:Unity-Technologies/ml-agents.git`
`cd ml-agents/ml-agents-envs`
`python3 setup.py build` (maybe followed by `python3 setup.py install`)
`cp -r cp -r mlagents/envs/ /usr/local/lib/python3.6/dist-packages/mlagents/`

where `/usr/local/lib/python3.6/dist-packages/mlagents/` may be different in your system, and you can find out what it is by running `import mlagents; print(mlagents.__path__)` in python3.

I am using the unreleased master-branch version of mlagents_envs because the documentation that is currently being written is for this version, plus it has a new neat feature (side_channels) which allow to set extra information to/from Unity (other than Observations/Actions).

See new jupyter notebook [here](https://github.com/Unity-Technologies/ml-agents/blob/master/notebooks/getting-started.ipynb), the new API documentation [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Python-API.md). There's also a Gym wrapper [here](https://github.com/Unity-Technologies/ml-agents/blob/master/gym-unity/README.md), but I haven't tested it.

PS also make sure to use Python 3.6 as Tensorflow doesn't work or work well with 3.7 or 3.8.
