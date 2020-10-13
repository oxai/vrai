using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Grpc.Core;
using FrooxEngine.LogiX;
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
            node.Debug.Log(node.GetHashCode());
        }

        public override Task<NeosObservation> GetObs(Empty f, ServerCallContext context)
        {
            return Task.FromResult(new NeosObservation
            {
                Obs = {node.obs_tmp}, SideInfo = { node.side_info_tmp }, Reward = node.reward_tmp, ShouldReset = node.should_reset
            });
        }

        public override Task<Response> SendAct(NeosAction action, ServerCallContext context)
        {
            try
            {
                action.Action.CopyTo(node.actions_tmp, 0);
                node.MLAgentsUpdateEvent.Set();
                //We block here for Neos to complete performing the action, and gathering the new observations, to keep sync!
                node.NeosUpdateEvent.Wait();
                node.NeosUpdateEvent.Reset();
                return Task.FromResult(new Response { Res = "Ok" });
            }
            catch (Exception exception)
            {
                var error = "Server threw exeception at SendAct : " + exception.Message;
                return Task.FromResult(new Response{ Res = error});
            }

        }

        public override Task<NeosAction> GatherAct(Empty f, ServerCallContext context)
        {
            try
            {
                //node.MLAgentsUpdateEvent.Set();
                //imitation learning synchronization assumes that neos is slower than unity hmm
                //because we are not blocking the actual movement of the user-controlled agent
                //this is fine if we use GAIL to imitate state sequences only, I think
                //node.NeosUpdateEvent.Wait();
                //node.NeosUpdateEvent.Reset();
                return Task.FromResult(new NeosAction
                {
                    Action = { node.demo_actions_tmp },
                });
            }
            catch (Exception exception)
            {
                var error = "Server threw exeception at GatherAct : " + exception.Message;
                node.Debug.Log(error);
                return Task.FromResult(new NeosAction
                {
                    Action = { node.demo_actions_tmp },
                });
            }
        }

        public override Task<TextureObservation> GetTextureObs(Empty f, ServerCallContext context)
        {
            try
            {
                ByteString[] raw_textures = new ByteString[node.textures.Length];
                for (int i = 0; i < node.textures.Length; i++)
                {
                    raw_textures[i] = ByteString.CopyFrom(node.textures[i]);
                }
                return Task.FromResult(new TextureObservation
                {
                    Textures = {raw_textures}
                }) ;
            }
            catch (Exception exception)
            {
                var error = "Server threw exeception at GetTextureObs : " + exception.Message;
                node.Debug.Log(error);
                ByteString[] raw_textures = new ByteString[node.textures.Length];
                for (int i = 0; i < node.textures.Length; i++)
                {
                    raw_textures[i] = ByteString.CopyFrom(node.textures[i]);
                }
                return Task.FromResult(new TextureObservation
                {
                    Textures = {raw_textures}
                }) ;
            }
        }

        public override Task<Response> ResetAgent(BareObs f, ServerCallContext context)
        {

            try
            {
                node.MLAgentsUpdateEvent.Set();
                node.should_reset = false;
                node.have_reset = true;
                node.NeosUpdateEvent.Wait();
                node.NeosUpdateEvent.Reset();
                f.Obs.CopyTo(node.reset_obs_tmp, 0);
                return Task.FromResult(new Response { Res = "Ok" });
            }
            catch (Exception exception)
            {
                var error = "Server threw exeception at ResetAgent : " + exception.Message;
                node.Debug.Log(error);
                return Task.FromResult(new Response{ Res = error});
            }
        }
        public override Task<ConnectionConfig> EstablishConnection(ConnectionParams parameters, ServerCallContext context)
        {

            try
            {
                node.Debug.Log("Setting connected_to_ml_agents to true");
                node.Debug.Log(node.connected_to_mlagents);
                node.have_read = true;
                node.connected_to_mlagents = true;
                node.Debug.Log(node.connected_to_mlagents);
                if (parameters.IsRecording)
                    node.recording_demo_tmp = parameters.IsRecording;
                Response res = new Response { Res = "Ok" };
                return Task.FromResult(new ConnectionConfig { ActionDim = node.action_dim,
                                                              ObsDim = node.obs_dim,
                                                              DoRecording = node.recording_demo_tmp,
                                                              AgentIndex = node.copy_idx_tmp,
                                                              NumberAgents = node.number_agents_tmp,
                                                              DemoFile = node.demo_file_tmp,
                                                              Res = res,
                                                              VisObsDim = node.vis_obs_dim});
            }
            catch (Exception exception)
            {
                var error = "Server threw exeception at EstablishConnection : " + exception.Message;
                node.Debug.Log(error);
                Response res = new Response { Res = error };
                return Task.FromResult(new ConnectionConfig { ActionDim = node.action_dim,
                                                              ObsDim = node.obs_dim,
                                                              DoRecording = node.recording_demo_tmp,
                                                              AgentIndex = node.copy_idx_tmp,
                                                              NumberAgents = node.number_agents_tmp,
                                                              Res = res,
                                                              VisObsDim = node.vis_obs_dim});
                }
        }
        public override Task<Response> StopConnection(Empty f, ServerCallContext context)
        {
            try
            {
                node.connected_to_mlagents = false;
                node.recording_demo_tmp = false;
                node.MLAgentsUpdateEvent.Set();
                return Task.FromResult(new Response { Res = "Ok" });
            }
            catch (Exception exception)
            {
                var error = "Server threw exeception at StopConnection : " + exception.Message;
                return Task.FromResult(new Response{ Res = error});
            }
        }
    }
}
