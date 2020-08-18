---
layout: default
---

# Initial experiments

At the beginning of the project, most of us had little experience (at least hands-on experience) with reinforcement learning (RL), so we played around with ML-Agents and custom RL code on simple tasks to get experience, even before the software framework to connect to VR was ready.

## Curiosity-driven learning

## Hide and Seek

A simple multi-agent RL task, using ML-Agent

## Imitation learning vs RL

... We decided to focus on IL

# VR experiments

After the [software framework](software) for connecting ML-Agents and NeosVR was ready, we began doing some simple experiments using VR demonstration data.

## Alphali

The first experiment looked at a simple agent with low-dimensional action space, but a high dimensional visual observation. We called this agent alphali combining the fact that we used an "avali" avatar as the body of the agent, and that it was the first (alpha). The task consisted of moving around until you saw a red virus in the scene, and then run to get to it. The virus would respwan in a random place in the arena with a certain (randomized) frequency). The agent could only rotate around the vertical axis, and decide its forward/backward velocity, resulting in a 2-dimensional action space. The observations came from one 84x84 pixel camera looking forward from the head of the agent.

An example of me recording a demo in VR is shown in the following video

...video

After training for a few hours, the agent began showing the right behaviour:

...video

Before obtaining these results, we had to fix several bugs, including a serious one whereby the camera observation was blank. However, the final results looked promising, and we decided to move to the next challenge: expanding the action space to include more degrees of freedom. I (Guillermo) decided to try the full 18-dimensional degrees of freedom corresponding to position and rotation of both hands and head, which I called the betali. Ryan explored a simpler agent which could just move one hand, to try to see if it could learn to wave, so he called it the wavili. These experiments are described next

## betali

## Wavili
