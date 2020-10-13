using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Grpc.Core;
using System;
using MLAgents;
using System.Threading;


public class VoidEnvSettingsOverrides : MonoBehaviour
{
    [Tooltip("This is the prototype agent to clone")]
    public GameObject ProtoAgent;
    public void Awake()
    {
        try
        {
            var channel = new Channel("127.0.0.1:40052", ChannelCredentials.Insecure);
            var client = new DataComm.DataCommClient(channel);
            var response = client.EstablishConnection(new ConnectionParams { IsRecording = false });
            int number_agents = response.NumberAgents;
            Debug.Log("Number of agents: " + number_agents.ToString());
            for (int i = 1; i < number_agents; i++)
            {
                Debug.Log("Creating agent copy " + i.ToString());
                GameObject new_copy = UnityEngine.Object.Instantiate(ProtoAgent);
                new_copy.GetComponent<TestAgent>().agent_index = i;
                //Thread.Sleep(300);
                new_copy.GetComponent<TestAgent>().Initialize();
            }
            var res = client.StopConnection(new Empty());
            if (res.Res != "Ok")
                Debug.Log(res.Res);
        }
        catch (Exception e)
        {
            Debug.Log("Exception caught in VoidEnvSettingsOverrides" + e.ToString());
        }
    }
    public void OnDestroy()
    {

    }

}
