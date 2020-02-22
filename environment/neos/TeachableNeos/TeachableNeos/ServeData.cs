using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BaseX;
using FrooxEngine;
//using FrooxEngine.LogiX;

//using Grpc.Core;
//using Grpc.Core.Utils;
//using LogiX;
using FrooxEngine.LogiX;
using Grpc.Core;

namespace FrooxEngine.LogiX.Network
{
    [Category("LogiX/Network")]
    [NodeName("Serve Data")]
    public class ServeData : LogixNode
    {
        public readonly Input<float> TestNumber;
        public readonly Input<float> TestNumber2;

        public readonly Output<int> TestOutput;
        public DataComm.DataCommClient client;
        public Channel channel;

        public override void RunStartup()
        {
            base.RunStartup();
            StartRPCServer();
            //RunInBackground(() =>
            //{
            //    StartRPCServer();
            //})
        }

        //protected override void OnChanges()
        //{

        //}

            private void StartRPCServer()
        {
            channel = new Channel("127.0.0.1:50052", ChannelCredentials.Insecure);
            this.client = new DataComm.DataCommClient(channel);

        }

        protected override void OnEvaluate()
        {
            var number1 = TestNumber.EvaluateRaw();
            var number2 = TestNumber2.EvaluateRaw();
            Feature request = new Feature { Thing = number1, Thing2 = number2 };
            Classification c = this.client.SendFeatures(request);
            int k = c.K;
            TestOutput.Value = k;
        }

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
            channel.ShutdownAsync().Wait();
        }
    }
}
