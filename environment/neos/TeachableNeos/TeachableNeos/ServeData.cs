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

namespace FrooxEngine.LogiX.Network
{
    [Category("LogiX/Network")]
    [NodeName("Learn")]
    public class ServeData : LogixNode
    {
        public readonly Input<float> x;
        public float x_tmp;
        public readonly Input<float> y;
        public float y_tmp;
        public readonly Input<float> z;
        public float z_tmp;
        public readonly Input<float> reward;
        public float reward_tmp;
        public readonly Input<bool> recording_demo;
        public readonly Input<int> copy_idx;
        public int copy_idx_tmp;
        public readonly Impulse Pulse;
        public readonly Impulse ResetPulse;
        public byte[] texture;
        public bool should_reset = false;

        public readonly Input<Camera> camera;

        public readonly Input<float> body_vx_human;
        public readonly Input<float> body_vz_human;
        public readonly Input<float> body_wy_human;
        public float body_vx_human_tmp;
        public float body_vz_human_tmp;
        public float body_wy_human_tmp;
        //public readonly Output<int> TestOutput;
        public readonly Output<float> body_vx;
        public readonly Output<float> body_vz;
        public readonly Output<float> body_wy;
        //public readonly Sync<float> body_vx;
        //public readonly Sync<float> body_vz;
        //public readonly Sync<float> body_wy;

        public float body_vx_tmp;
        public float body_vz_tmp;
        public float body_wy_tmp;
        public bool have_read;
        public bool have_reset;
        //private DataComm.DataCommClient client;
        public Channel channel;
        public Server server;

        public override void RunStartup()
        {
            base.RunStartup();
            //StartRPCServer();
            StartRPCServer();
        }

        private void StartRPCServer()
        {
            Task.Run(()=> {
                try
                {
                    //channel = new Channel("127.0.0.1:50052", ChannelCredentials.Insecure);
                    //this.client = new DataComm.DataCommClient(channel);
                    int Port = 50052+copy_idx_tmp;

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

        public void UpdateObservations()
        {
                x_tmp = this.x.Evaluate();
                y_tmp = this.y.Evaluate();
                z_tmp = this.z.Evaluate();
                reward_tmp = this.reward.Evaluate();
                body_vx_human_tmp = this.body_vx_human.Evaluate();
                body_vz_human_tmp = this.body_vz_human.Evaluate();
                body_wy_human_tmp = this.body_wy_human.Evaluate();
        }
        [ImpulseTarget]
        public void ResetAgent()
        {
            should_reset = true;
        }

        [ImpulseTarget]
        public void PerformAction()
        {
            body_vx.Value = this.body_vx_tmp;
            body_vz.Value = this.body_vz_tmp;
            body_wy.Value = this.body_wy_tmp;
            if (have_read)
            {
                if (!recording_demo.Evaluate())
                {
                    Pulse.Trigger();
                }
                have_read = false;
            }
            if (have_reset)
            {
                ResetPulse.Trigger();
                have_reset = false;
                UpdateObservations();
                body_vx_human_tmp = 0;
                body_vz_human_tmp = 0;
                body_wy_human_tmp = 0;
            } else
            {
                UpdateObservations();
            }
            copy_idx_tmp = this.copy_idx.Evaluate();
            
            //texture = RenderManager.RenderToBitmap(camera.Evaluate().GetRenderSettings(new int2(84,84))).Wait().RawData();
            texture = base.World.Render.Connector.Render(camera.Evaluate().GetRenderSettings(new int2(84,84)));
            //Debug.Log(texture[1].ToString());
        }

        [ImpulseTarget]
        public void reset_server()
        {
                server.ShutdownAsync().Wait();
                StartRPCServer();
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

        //protected override void OnEvaluate()
        //{
        //    var number1 = TestNumber.EvaluateRaw();
        //    var number2 = TestNumber2.EvaluateRaw();
        //    //Feature request = new Feature { Thing = number1, Thing2 = number2 };
        //    //Classification c = this.client.SendFeatures(request);
        //    //int k = c.K;
        //    TestOutput.Value = 1;
        //}

            //protected override void InitializeSyncMembers()
            //{
            //    base.InitializeSyncMembers();
            //    this.TestNumber = new Input<float>();
            //    this.TestNumber2 = new Input<float>();
            //    this.TestOutput = new Output<float>();
            //}


            //public override ISyncMember GetSyncMember(int index)
            //{
            //    switch (index)
            //    {
            //        case 0:
            //            return (ISyncMember)this.persistent;
            //        case 1:
            //            return (ISyncMember)this.updateOrder;
            //        case 2:
            //            return (ISyncMember)this.EnabledField;
            //        case 3:
            //            return (ISyncMember)this._activeVisual;
            //        case 4:
            //            return (ISyncMember)this.TestNumber;
            //        case 5:
            //            return (ISyncMember)this.TestNumber2;
            //        case 6:
            //            return (ISyncMember)this.TestOutput;
            //        default:
            //            throw new ArgumentOutOfRangeException();
            //    }
            //}

            //public static ServeData __New()
            //{
            //    return new ServeData();
            //}

            //protected override void NotifyOutputsOfChange()
            //{
            //    ((IOutputElement)this.TestOutput).NotifyChange();
            //}

            //[ImpulseTarget]
            //public void Run()
            //{

            //}

        public override void RunOnDestroying()
        {
            base.RunOnDestroying();
            //channel.ShutdownAsync().Wait();
            server.ShutdownAsync().Wait();
        }
    }
    //[OldNamespace("FrooxEngine")]
    //[NodeName("hellothere")]
    //[Category("LogiX/Network")]
    //public class Dec_Int : LogixOperator<int>
    //{
    //    public readonly Input<int> A;

    //    public override int Content { get { return (int)(A.EvaluateRaw() - 1); } }
    //}

}

