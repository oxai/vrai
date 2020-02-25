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

public class TestAgent : Agent
{
    private DataComm.DataCommClient client;
    public TestAcademy academy;
    public List<float> inputs; //can't call this variable observations, coz I guess that's being used for something else? dunno
    public float stop_training = 0;
    Texture2D tex;
    RawImage image;
    public override void InitializeAgent()
    {
        //inputs = new List<float>() {0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f};
        academy = FindObjectOfType<TestAcademy>();
        client = academy.client;
        //using (var client = new TcpClient<IVrcaiMlaTest>(ipEndPoint))
        //{
        //    TextureMessage response = client.Proxy.getObs(new List<float>{0f,0f,0f});
        //    image = GameObject.Find("RawImage").GetComponent<RawImage>();
        //    //image.uvRect = new Rect(0,0, response.width, response.height);
        //    GameObject.Find("RawImage").GetComponent<RectTransform>().sizeDelta = new Vector2(response.width,response.height);
        //    //GameObject.Find("RawImage").GetComponent<RectTransform>().
        //    //Debug.Log("hi");
        //    //rt = new RenderTexture(tex.width / 2, tex.height / 2, 0);
        //}
        //m_BallRb = ball.GetComponent<Rigidbody>();
        //var academy = FindObjectOfType<Academy>();
        //m_ResetParams = academy.FloatProperties;
        //SetResetParameters();
    }

    public override void CollectObservations()
    {
        NeosObservation obs = client.GetObs(new Empty());
        //Debug.Log(obs.X);
        //Debug.Log(obs.Z);
        //Debug.Log(inputs.Count);
        AddVectorObs(obs.X);
        AddVectorObs(obs.Z);
        float reward = obs.Reward;
        //Debug.Log("Reward " + reward.ToString());
        AddReward(reward);
        stop_training = 0f;
    }

    //void FixedUpdate()
    //{

    //    using (var client = new TcpClient<IVrcaiMlaTest>(ipEndPoint))
    //    {
    //        //inputs = client.Proxy.getObs(new List<float>(vectorAction));
    //        TextureMessage response = client.Proxy.getObs(new List<float>(new List<float> { 0f, 0f, 0f }));
    //        //Debug.Log(response.raw_texture[0]);
    //        Destroy(tex);
    //        tex = new Texture2D(response.width, response.height, TextureFormat.RGB24, false);
    //        tex.LoadRawTextureData(response.raw_texture);
    //        tex.Apply();
    //        //Debug.Log(image);
    //        image.texture = tex;
    //    }
    //}

    public override void AgentAction(float[] vectorAction)
    {

        if (GetStepCount() >= 100)
        {
            Response res = client.SendAct(new NeosAction { BodyVx = vectorAction[0], BodyVz = vectorAction[1] });
            //Response res = client.SendAct(new NeosAction { BodyVx = 0.1f, BodyVz = 0.5f });
            Debug.Log(res.Res);
            Done();
        } else
        {
            Response res = client.SendAct(new NeosAction { BodyVx = vectorAction[0], BodyVz = vectorAction[1] });
            //Response res = client.SendAct(new NeosAction { BodyVx = 0.1f, BodyVz = 0.5f });
            Debug.Log(res.Res);
        }

    }

    public override void AgentReset()
    {
        Response res = client.ResetAgent(new Empty());
        //Response res = client.SendAct(new NeosAction { BodyVx = 0.1f, BodyVz = 0.5f });
        Debug.Log(res.Res);
    }

    public void Update()
    {
    }

    public override float[] Heuristic()
    {
        var action = new float[2];
        return action;
    }

}

