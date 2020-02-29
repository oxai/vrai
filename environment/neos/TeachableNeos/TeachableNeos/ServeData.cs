using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BaseX;
using FrooxEngine;
using FrooxEngine.LogiX;
using Grpc.Core;

namespace FrooxEngine.LogiX.Network
{
    [Category("LogiX/Network")]
    [NodeName("Serve Data")]
    public class ServeData : LogixNode
    {
        public readonly Input<float> x;
        public float x_tmp;
        public readonly Input<float> z;
        public float z_tmp;
        public readonly Input<float> reward;
        public float reward_tmp;
        public readonly Impulse Pulse;
        public readonly Impulse ResetPulse;

        public readonly Input<float> body_vx_human;
        public readonly Input<float> body_vz_human;
        public float body_vx_human_tmp;
        public float body_vz_human_tmp;
        //public readonly Output<int> TestOutput;
        //public readonly Output<float> body_vx;
        //public readonly Output<float> body_vy;
        public readonly Sync<float> body_vx;
        public readonly Sync<float> body_vz;

        public float body_vx_tmp;
        public float body_vz_tmp;
        public bool have_read;
        public bool have_reset;
        //private DataComm.DataCommClient client;
        public Channel channel;
        public Server server;

        public override void RunStartup()
        {
            base.RunStartup();
            //StartRPCServer();
            Task.Run(()=> {
                try
                {
                    StartRPCServer();
                }
                catch (Exception exception)
                {
                    Debug.Log("Server threw exeception : " + exception.Message);
                }
            });
        }

        //public void Trigger(NeosAction action)
        //{
        //    //body_vx.Value = action.BodyVx;
        //    //body_vy.Value = action.BodyVz;
        //    //Pulse.Trigger();
        //}

        //protected override void OnChanges()
        //{

        //}
        //protected override void InitializeSyncMembers()
        //{
        //    base.InitializeSyncMembers();
        //    this.Pulse = new Impulse();
        //    this.body_vx = new Output<float>();
        //    this.body_vz = new Output<float>();
        //}

        private void StartRPCServer()
        {
            //channel = new Channel("127.0.0.1:50052", ChannelCredentials.Insecure);
            //this.client = new DataComm.DataCommClient(channel);
            const int Port = 50052;

            server = new Server
            {
                Services = { DataComm.BindService(new TeachableNeos.DataCommImpl(this)) },
                Ports = { new ServerPort("127.0.0.1", Port, ServerCredentials.Insecure) }
            };
            server.Start();

        }

        [ImpulseTarget]
        public void SendAction()
        {
            x_tmp = this.x.Evaluate();
            z_tmp = this.z.Evaluate();
            reward_tmp = this.reward.Evaluate();
            body_vx_human_tmp = this.body_vx_human.Evaluate();
            body_vz_human_tmp = this.body_vz_human.Evaluate();
            body_vx.Value = this.body_vx_tmp;
            body_vz.Value = this.body_vz_tmp;
            if (have_read)
            {
                Pulse.Trigger();
                have_read = false;
            }
            if (have_reset)
            {
                ResetPulse.Trigger();
                have_reset = false;
            }
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
}
