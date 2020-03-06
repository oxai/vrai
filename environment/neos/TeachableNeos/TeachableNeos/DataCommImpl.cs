using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Grpc.Core;
using FrooxEngine.LogiX.Network;
using FrooxEngine;
using Google.Protobuf;

namespace TeachableNeos
{
    class DataCommImpl: DataComm.DataCommBase
    {
        public ServeData node;
        public DataCommImpl(ServeData parent_node)
        {
            node = parent_node;
        }

        public override Task<NeosObservation> GetObs(Empty f, ServerCallContext context)
        {
            return Task.FromResult(new NeosObservation
            {
                X = node.x_tmp, Y = node.y_tmp, Z = node.z_tmp, Reward = node.reward_tmp, ShouldReset = node.should_reset
            });
        }

        public override Task<Response> SendAct(NeosAction action, ServerCallContext context)
        {
            try
            {
                node.body_vx_tmp = action.BodyVx;
                node.body_vz_tmp = action.BodyVz;
                node.body_wy_tmp = action.BodyWy;
                node.have_read = true;
                //node.body_vx = default(float);
                //node.body_vy = default(float);
                return Task.FromResult(new Response { Res = "Ok" });
            }
            catch (Exception exception)
            {
                var error = "Server threw exeception : " + exception.Message;
                return Task.FromResult(new Response{ Res = error});
            }
        }

        public override Task<NeosAction> GatherAct(Empty f, ServerCallContext context)
        {
            return Task.FromResult(new NeosAction
            {
                BodyVx = node.body_vx_human_tmp,
                BodyVz = node.body_vz_human_tmp,
                BodyWy = node.body_wy_human_tmp
            });
        }

        public override Task<TextureObservation> GetTextureObs(Empty f, ServerCallContext context)
        {
            try
            {
                return Task.FromResult(new TextureObservation
                {
                    Texture = ByteString.CopyFrom(node.texture)
                }) ;
            }
            catch (Exception exception)
            {
                var error = "Server threw exeception : " + exception.Message;
                node.Debug.Log(error);
                return Task.FromResult(new TextureObservation
                {
                    Texture = ByteString.CopyFrom(node.texture)
                }) ;
            }
        }

        public override Task<Response> ResetAgent(Empty f, ServerCallContext context)
        {

            try
            {
                node.should_reset = false;
                node.have_reset = true;
                return Task.FromResult(new Response { Res = "Ok" });
            }
            catch (Exception exception)
            {
                var error = "Server threw exeception : " + exception.Message;
                return Task.FromResult(new Response{ Res = error});
            }
        }
    }
}
