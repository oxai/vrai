using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BaseX;
using FrooxEngine;
using FrooxEngine.LogiX;
using Grpc.Core;
//using CodeX;
//using UnityEngine;
using FrooxEngine.LogiX;
using System.Threading;
using System.Linq;
using FrooxEngine.UIX;

namespace FrooxEngine.LogiX
{
    [Category("LogiX/VRAI")]
    [NodeName("Learn")]
    public class ServeData : LogixNode
    {
        public readonly Input<float[]> obs;
        public float[] obs_tmp = new float[0];
        public readonly Input<float[]> side_info;
        public float[] side_info_tmp = new float[0];
        public readonly Input<float> reward;
        public float reward_tmp = 0.0f;
        public readonly Input<Camera[]> cameras;
        public readonly Input<int> copy_idx;
        public int copy_idx_tmp = 0;
        public readonly Input<bool> record_demo;
        public bool recording_demo_tmp = false;
        public readonly Input<float[]> demo_actions;
        public float[] demo_actions_tmp = new float[0];
        public readonly Input<int> number_agents;
        public int number_agents_tmp = 1;
        public readonly Input<string> demo_file;
        public string demo_file_tmp = default(string);

        //Outputs
        public readonly Impulse DoAction;
        public readonly Impulse ResetAgent;
        public readonly Output<float[]> actions;
        public readonly Output<float[]> reset_obs;
        //public readonly Sync<float[]> actions;
        public float[] actions_tmp = new float[0];

        //Internal variables
        public int action_dim, obs_dim, vis_obs_dim;
        public byte[][] textures = new byte[0][];
        public bool should_reset = false;
        public bool connected_to_mlagents = false;
        public bool have_read;
        public bool has_updated;
        public bool have_reset;
        public float[] reset_obs_tmp = new float[0];
        //private DataComm.DataCommClient client;
        public Channel channel;
        public Server server;
        public ManualResetEventSlim NeosUpdateEvent = new ManualResetEventSlim(false);
        public ManualResetEventSlim MLAgentsUpdateEvent = new ManualResetEventSlim(false);

        protected override void OnStart()
        {
            Debug.Log("starting Learn node!");
            base.OnStart();
            copy_idx_tmp = this.copy_idx.Evaluate();
            //StartRPCServer();
            StartRPCServer();
            //actions.Value = new float[0];
        }

        private void StartRPCServer()
        {
            Task.Run(() =>
            {
                try
                {
                    //channel = new Channel("127.0.0.1:50052", ChannelCredentials.Insecure);
                    //this.client = new DataComm.DataCommClient(channel);
                    int Port = 50052 + copy_idx_tmp;
                    Debug.Log("Hiiiia, starting server at " + Port.ToString());
                    GrpcEnvironment.SetLogger(new Grpc.Core.Logging.ConsoleLogger());

                    server = new Server
                    {
                        Services = { DataComm.BindService(new TeachableNeos.DataCommImpl(this)) },
                        Ports = { new ServerPort("127.0.0.1", Port, ServerCredentials.Insecure) }
                    };
                    server.Start();
                }
                catch (Exception exception)
                {
                    Debug.Log("Server threw exeception : " + exception.Message);
                    Debug.Log("Server threw exeception : " + exception.ToString());
                    Debug.Log("Server threw exeception : " + exception.InnerException.Message);
                    Debug.Log("Server threw exeception : " + exception.InnerException.InnerException.Message);
                }
            });

        }


        public void CollectInputs()
        {
            //vector obs
            obs_tmp = this.obs.EvaluateRaw(new float[0]);
            //side info
            side_info_tmp = this.side_info.EvaluateRaw(new float[0]);
            //reward
            reward_tmp = this.reward.Evaluate(default);
            //demo actions
            demo_actions_tmp = this.demo_actions.EvaluateRaw(new float[0]);
            //demo_file_tmp
            demo_file_tmp = this.demo_file.EvaluateRaw(default(string));

            //agent index and other brain configs
            copy_idx_tmp = this.copy_idx.Evaluate(default);
            number_agents_tmp = this.number_agents.Evaluate(default);
            recording_demo_tmp = this.record_demo.Evaluate(default);
            action_dim = demo_actions_tmp.Length;
            if (action_dim != actions_tmp.Length) actions_tmp = new float[action_dim];
            obs_dim = obs_tmp.Length;
            if (obs_dim != reset_obs_tmp.Length) reset_obs_tmp = new float[obs_dim];

            //visual inputs
            var cameras_evald = cameras.EvaluateRaw(new Camera[0]);
            int num_cameras = cameras_evald.Length;
            vis_obs_dim = num_cameras;
            if (num_cameras != textures.Length) textures = new byte[num_cameras][];
            if (Engine.SystemInfo.HeadDevice != HeadOutputDevice.Headless)
            {
                for (int i = 0; i < num_cameras; i++)
                {
                    textures[i] = base.World.Render.Connector.Render(cameras_evald[i].GetRenderSettings(new int2(84, 84)));
                }
            }
            //texture = RenderManager.RenderToBitmap(camera.Evaluate().GetRenderSettings(new int2(84, 84))).Wait().RawData();
            //texture = base.World.Render.Connector.Render(camera.Evaluate().GetRenderSettings(new int2(84, 84)));
        }

        [ImpulseTarget]
        public void PerformAction()
        {
            //wait to receive action from MLAgents
            if (connected_to_mlagents)
            {
                MLAgentsUpdateEvent.Wait();
                MLAgentsUpdateEvent.Reset();
                //update output to the latest action received
                actions.Value = this.actions_tmp;
                reset_obs.Value = this.reset_obs_tmp;
                //if not recording then we perform the action
                if (!recording_demo_tmp)
                {
                    DoAction.Trigger();
                }

                //if MLAgents said we have to reset, then we send the reset trigger, and update inputs (but reset demo_actions, coz the agent may have jumped by the reset operation)
                if (have_reset)
                {
                    ResetAgent.Trigger();
                    have_reset = false;
                    CollectInputs();
                    Array.Clear(demo_actions_tmp, 0, demo_actions_tmp.Length);
                }
                else //otherwise, we just collect the new inputs
                {
                    CollectInputs();
                }
            }
            else
            {
                CollectInputs(); //this will update action_dim
                //update output to the latest action received
                //if (actions.Value.Length != action_dim) actions.Value = new float[action_dim];
                actions.Value = new float[action_dim];
            }


            //Tell DataComm, that Neos has finished running (and gathering new obs/demo_actions)
            if (connected_to_mlagents)
            {
                NeosUpdateEvent.Set();
            }
        }

        [ImpulseTarget]
        public void SetAgentToReset()
        {
            CollectInputs();
            should_reset = true;
        }

        [ImpulseTarget]
        public void StartServer()
        {
            CollectInputs();
            StartRPCServer();
        }

        [ImpulseTarget]
        public void StopServer()
        {
            CollectInputs();
            server.ShutdownAsync().Wait();
        }

        protected override void OnCommonUpdate()
        {
            //Pass the callback to the base so the output are updated
            base.OnCommonUpdate();
        }

        protected override void OnChanges()
        {
            //Pass the callback to the base so the outputs are updated on the node
            base.OnChanges();
        }

        public override void RunOnDestroying()
        {
            base.RunOnDestroying();
            //channel.ShutdownAsync().Wait();
            server.ShutdownAsync().Wait();
        }
    }
}
