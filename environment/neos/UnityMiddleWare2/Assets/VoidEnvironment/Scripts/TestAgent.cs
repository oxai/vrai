using UnityEngine;
using MLAgents;
using System.Collections.Generic;
using System.Net;
using System;
using System.Drawing;
//using System.Windows.Forms;
using System.Diagnostics;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEngine.UI;
using Debug = UnityEngine.Debug;
using Screen = System.Windows.Forms.Screen;
using Google.Protobuf;
using UnityEngine.UI;
using Grpc.Core;

public class TestAgent : Agent
{
    private DataComm.DataCommClient client;
    public TestAcademy academy;
    //public List<float> inputs; //can't call this variable observations, coz I guess that's being used for something else? dunno
    public float stop_training = 0;
    public bool should_reset = false;
    public int texture_width = 84, texture_height = 84;
    public int agent_index = 0;
    bool is_recording = false;
    Texture2D tex;
    RawImage image;
    public Camera dummyCamera;
    public RawImage raw_image;
    private ConnectToNeos cameraToNeos;

    public override void InitializeAgent()
    {
        var channel = new Channel("127.0.0.1:5005"+(2+agent_index).ToString(), ChannelCredentials.Insecure);
        this.client = new DataComm.DataCommClient(channel);
        is_recording =this.GetComponent<DemonstrationRecorder>().record;
        cameraToNeos = dummyCamera.GetComponent<ConnectToNeos>();
        image = raw_image.GetComponent<RawImage>();
        var res = client.EstablishConnection(new ConnectionParams { IsRecording = is_recording });
        if (res.Res != "Ok")
            Debug.Log(res.Res);
    }

    public override void CollectObservations()
    {
        NeosObservation obs = client.GetObs(new Empty());
        //Debug.Log("X: "+obs.X.ToString());
        //Debug.Log("Z: "+obs.Z.ToString());
        //Debug.Log(inputs.Count);
        AddVectorObs(obs.X);
        AddVectorObs(obs.Z);
        float reward = obs.Reward;
        //Debug.Log("Reward " + reward.ToString());
        AddReward(reward);
        //stop_training = 0f;
        should_reset = obs.ShouldReset;
        TextureObservation res = client.GetTextureObs(new Empty());
        byte[] texture_bytes = res.Texture.ToByteArray();
        //Debug.Log("Length of texture bytes: "+texture_bytes.Length.ToString());
        Destroy(tex);
        tex = new Texture2D(texture_width, texture_height, TextureFormat.ARGB32, false);
        tex.LoadRawTextureData(texture_bytes);
        tex.Apply();
        image.texture = tex;
        cameraToNeos.texture = tex;

    }

    public override void AgentAction(float[] vectorAction)
    {
        Debug.Log("vx: " + vectorAction[0].ToString());
        Debug.Log("vz: " + vectorAction[1].ToString());
        Debug.Log("wy: " + vectorAction[2].ToString());
        Response res;
        //if (should_reset || GetStepCount() >= 2500)
        if (should_reset || GetStepCount() >= 25000)
        {
            res = client.SendAct(new NeosAction { BodyVx = vectorAction[0], BodyVz = vectorAction[1], BodyWy = vectorAction[2] });
            if (res.Res != "Ok")
                Debug.Log(res.Res);
            Done();
        } else
        {
            res = client.SendAct(new NeosAction { BodyVx = vectorAction[0], BodyVz = vectorAction[1], BodyWy = vectorAction[2] });
            if (res.Res != "Ok")
                Debug.Log(res.Res);
        }

    }

    public override void AgentReset()
    {
        Response res = client.ResetAgent(new Empty());
        if (res.Res != "Ok")
            Debug.Log(res.Res);
    }

    public void Update()
    {
    }

    public override float[] Heuristic()
    {
        NeosAction action_message = client.GatherAct(new Empty());
        var action = new float[3];
        action[0] = action_message.BodyVx;
        action[1] = action_message.BodyVz;
        action[2] = action_message.BodyWy;
        //Debug.Log(action[0]);
        return action;
    }

    private void OnApplicationQuit()
    {
        var res = client.StopConnection(new Empty());
        if (res.Res != "Ok")
            Debug.Log(res.Res);
    }

}

