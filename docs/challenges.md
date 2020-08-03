---
layout: default
---

Introduction

# VR platform and data collection

To accomplish our goals, we need to use a multi-player VR software. To maximize the potential for data gathering, we decided to look at existing social VR games that already had many users. The top "general" social VR games as of end of 2019 were (ignoring multi-player games with specific theme, like shooting games):

1. VRChat (~6000 concurrent users)
2. Rec Room (~200 concurrent users)
3. NeosVR (~20 concurrent users)
4. AltSpace (~5 concurrent users)

Several factors were at play when choosing a platform:

* Number of users. The more users, the more potential to scale the data collection
* Number of users using VR and full-body tracking. Most of these platforms allow for both VR and desktop users to play. We classify users into three rough categories, depending on the richness of the data available: desktop, VR, full-body (from less to more rich). We prefer platforms were there are more users with VR and full-body tracking.
* Ease of integration with custom software and creation of custom environments. This is necessary to do our AI experiments, which will rely on external ML software, and require custom experiments and data collection pipelines.

Out of the different candidates above, we decided to focus on VRChat because of the much higher number of users. This was at the expense of higher technical difficulty, as VRChat doesn't offer a nice API to integrate with external software. As we will discuss in the [software section](software), legal issues meant our approach for using VRChat couldn't be used, and we decided to pivot to NeosVR, as this was the platform where integration with external software would be easiest, the fraction of VR and full-body users was high, and the number of users showed a clear growing trend.

# ML framework/Software

# Connecting ML to VR

The main loop of reinforcement learning algorithms require running the environment in which the agent is training. Therefore, to avoid the communication between the VR environment and the "brains" of the agent (living in the ML software), we decided to look for the fastest way to send data between two processes (the VR game and the ML library). We decided to use RPCs as they allow for two-way communication with efficient serialization of data, and thus reduced latency in the communication. We used the gRPC library, which is the same one that ML-Agent uses under the hood to communicate between Unity and a Tensorflow ML script.

The second challenge was on building either a RPC client or server that was integrated with the VR platform of choice. For VRChat this involved developing a custom mod, while for Neos, we had to develop a custom plugin. This involved learning about C#, .NET, visual studio, Unity, etc. which we weren't familiar with, as well as the specifics of the platform.

ML-Agents is designed to communicate (via RPC) with a Unity application. To simplify our prototyping work, we decided to create a empty unity scene with a fake agent designed only to relay data between the VR application and ML-Agents. The details of this are in the [software section](software).

Finally, an extra challenge was keeping the ML-agent iterations in sync with the VR game's engine update step, which required fiddling with the plugin/mod depending on the platform.

# Cloud

# AI Challenges

Deciding on the types of algorithms we were going to use was a challenge. We knew we wanted to use human data, so that imitation learning was going to be an important ingredient, but we were also interested in curiosity-driven autonomous agents. However, from some of our initial [experiments](experiments) with pure RL, we decided to focus on imitation and supervised learning, as it produced more reliable results.

For simplicity, we decided to focus on the algorithms implemented in ML-Agents, which are GAIL with PPO and SAC, as well as simple behavioural cloning. Deciding the right hyperparameters and variations to these algorithms was, and still is currently, the biggest challenge in the project, as we will discuss in the [experiment](experiments) and [future work](future_work) sections.
