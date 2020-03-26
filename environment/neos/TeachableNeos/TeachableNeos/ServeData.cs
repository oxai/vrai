using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BaseX;
using FrooxEngine;
using FrooxEngine.LogiX;
using Grpc.Core;
using CodeX;
//using UnityEngine;
using FrooxEngine.LogiX;
using System.Threading;
using System.Linq;


namespace FrooxEngine.LogiX
{
    [Category("LogiX/VRAI")]
    [NodeName("Learn")]
    public class ServeData : LogixNode
    {
        public readonly Input<float[]> obs;
        public float[] obs_tmp;
        public readonly Input<float> reward;
        public float reward_tmp;
        public readonly Input<Camera[]> cameras;
        public readonly Input<int> copy_idx;
        public int copy_idx_tmp;
        public readonly Input<bool> record_demo;
        public bool recording_demo_tmp;
        public readonly Input<float[]> demo_actions;
        public float[] demo_actions_tmp;

        //Outputs
        public readonly Impulse DoAction;
        public readonly Impulse ResetAgent;
        public readonly Output<float[]> actions;
        //public readonly Sync<float[]> actions;
        public float[] actions_tmp;

        //Internal variables
        public int action_dim, obs_dim, vis_obs_dim;
        public byte[][] textures;
        public bool should_reset = false;
        public bool connected_to_mlagents = false;
        public bool have_read;
        public bool has_updated;
        public bool have_reset;
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
                    Debug.Log("Hiii, starting server at " + Port.ToString());

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
                }
            });

        }


        public void CollectInputs()
        {
            //vector obs
            obs_tmp = this.obs.Evaluate();
            //reward
            reward_tmp = this.reward.Evaluate();
            //demo actions
            demo_actions_tmp = this.demo_actions.Evaluate();

            //agent index and other brain configs
            copy_idx_tmp = this.copy_idx.EvaluateRaw();
            recording_demo_tmp = this.record_demo.EvaluateRaw();
            action_dim = demo_actions_tmp.Length;
            obs_dim = obs_tmp.Length;

            //visual inputs
            var cameras_evald = cameras.EvaluateRaw();
            int num_cameras = cameras_evald.Length;
            vis_obs_dim = num_cameras;
            if (num_cameras != textures.Length) textures = new byte[num_cameras][];
            for (int i = 0; i < num_cameras; i++)
            {
                textures[i] = base.World.Render.Connector.Render(cameras.EvaluateRaw()[i].GetRenderSettings(new int2(84, 84)));
            }
            //texture = RenderManager.RenderToBitmap(camera.Evaluate().GetRenderSettings(new int2(84,84))).Wait().RawData();
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
            }

            //update output to the latest action received
            actions.Value = this.actions_tmp;
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

            //Tell DataComm, that Neos has finished running (and gathering new obs/demo_actions)
            if (connected_to_mlagents)
            {
                NeosUpdateEvent.Set();
            }
        }

        [ImpulseTarget]
        public void SetAgentToReset()
        {
            should_reset = true;
        }

        [ImpulseTarget]
        public void StartServer()
        {
            StartRPCServer();
        }

        [ImpulseTarget]
        public void StopServer()
        {
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


    [OldNamespace("FrooxEngine")]
    [NodeName("To List")]
    [Category(new string[] { "LogiX/VRAI" })]
    //public class ToList : MultiInputOperator<Input<float>, List<float>>
    public class ToList : MultiInputOperator<float>
    {
        //public readonly List<float> Content
        public readonly Output<float[]> list;
        
        protected override void OnEvaluate()
        {
            base.OnEvaluate();
            list.Value = new float[this.Operands.Count];
            for (int index = 0; index < this.Operands.Count; ++index)
                list.Value[index] = this.Operands.GetElement(index).EvaluateRaw(default(float));
        }
        public override float Content
        {
            get
            {
                return default(float);
            }
        }
    }

    [OldNamespace("FrooxEngine")]
    [NodeName("Get Element")]
    [Category(new string[] { "LogiX/VRAI" })]
    public class GetElement: LogixNode
    {
        public readonly Input<float[]> list;
        public readonly Input<int> index;
        public readonly Output<float> element;

        protected override void OnEvaluate()
        {
            base.OnEvaluate();
            element.Value = list.EvaluateRaw()[index.EvaluateRaw()];
        }

    }

    //[OldNamespace("FrooxEngine")]
    //[NodeName("floatTofloat")]
    //[Category(new string[] { "LogiX/Operators" })]
    //[Category(new string[] { "Hidden" })]
    //public class FloatToFloatList : Cast.CastNode<float, float[]>, Cast.ICastNode, IPassthroughNode, IComponent, IComponentBase, IDestroyable, IWorker, IWorldElement, IUpdatable, IChangeable, IAudioUpdatable, IInitializable, ILinkable
    //{
    //    public override float[] Content
    //    {
    //        get
    //        {
    //            float[] result = { this.In.EvaluateRaw() };
    //            return result;
    //        }
    //    }
    //}


    //[OldNamespace("FrooxEngine")]
    //[NodeName("To List")]
    ////[NodeOverload("AddMulti")]
    ////[HiddenNode]
    //[Category(new string[] { "LogiX/Operators" })]
    //public class ListToString : LogixOperator<float>
    //{
    //    public readonly Input<int> Index;
    //    public readonly Input<float> list;
    //    public override float Content
    //    {
    //        get
    //        {
    //            //return list.EvaluateRaw()[Index.EvaluateRaw()];
    //            return list.EvaluateRaw();
    //        }
    //    }
    //}
}
