using UnityEngine;
using MLAgents;
using VrcaiMlaCommunicator;
using System.Collections.Generic;
using ServiceWire.TcpIp;
using System.Net;


public class TestAgent : Agent
{
    //[Header("Specific to Ball3D")]
    //public GameObject ball;
    //Rigidbody m_BallRb;
    //IFloatProperties m_ResetParams;
    public TcpClient<IVrcaiMlaTest> client;
    public IPEndPoint ipEndPoint;
    //public Academy academy;
    public List<float> inputs = new List<float> { 0, 0, 0 }; //can't call this variable observations, coz I guess that's being used for something else? dunno
    public float stop_training = 0;
    public override void InitializeAgent()
    {
        var academy = FindObjectOfType<TestAcademy>();
        client = academy.client;
        ipEndPoint = academy.ipEndPoint;
        //m_BallRb = ball.GetComponent<Rigidbody>();
        //var academy = FindObjectOfType<Academy>();
        //m_ResetParams = academy.FloatProperties;
        //SetResetParameters();
    }

    public override void CollectObservations()
    {
        //AddVectorObs(gameObject.transform.rotation.z);
        //AddVectorObs(gameObject.transform.rotation.x);
        //AddVectorObs(ball.transform.position - gameObject.transform.position);
        //AddVectorObs(m_BallRb.velocity);
        //AddVectorObs(1f);
        //Debug.Log(inputs.Count);
        //for (int i = 0; i < observations.Count; i++)
        //{
        //    Debug.Log(observations[i].ToString());
        //}
        for (int i = 0; i < inputs.Count-2; i++)
        {
            AddVectorObs(inputs[i]);
        }
        float reward = Mathf.Clamp(0.1f * inputs[inputs.Count - 2], -1f, 1f)+1f;
        Debug.Log("Reward " + reward.ToString());
        AddReward(reward);
        stop_training = inputs[inputs.Count - 1];
    }

    public override void AgentAction(float[] vectorAction)
    {
        //var actionZ = 2f * Mathf.Clamp(vectorAction[0], -1f, 1f);
        //var actionX = 2f * Mathf.Clamp(vectorAction[1], -1f, 1f);

        //if ((gameObject.transform.rotation.z < 0.25f && actionZ > 0f) ||
        //    (gameObject.transform.rotation.z > -0.25f && actionZ < 0f))
        //{
        //    gameObject.transform.Rotate(new Vector3(0, 0, 1), actionZ);
        //}

        //if ((gameObject.transform.rotation.x < 0.25f && actionX > 0f) ||
        //    (gameObject.transform.rotation.x > -0.25f && actionX < 0f))
        //{
        //    gameObject.transform.Rotate(new Vector3(1, 0, 0), actionX);
        //}
        //if ((ball.transform.position.y - gameObject.transform.position.y) < -2f ||
        //    Mathf.Abs(ball.transform.position.x - gameObject.transform.position.x) > 3f ||
        //    Mathf.Abs(ball.transform.position.z - gameObject.transform.position.z) > 3f)
        //{
        //    Done();
        //    SetReward(-1f);
        //}
        //else
        //{
        //    SetReward(0.1f);
        //}
        //List<float> actions = new List<float> { 0.1f, 0.1f, 0 };
        //observations = client.Proxy.getObs(new List<float>(vectorAction));
        //Debug.Log(vectorAction);
        if (GetStepCount() >= 2500 || stop_training == 1f)
        {
            using (var client = new TcpClient<IVrcaiMlaTest>(ipEndPoint))
                inputs = client.Proxy.getObs(new List<float>(vectorAction));
            //AddReward(-10f);
            Done();
        }
        else
        {
            using (var client = new TcpClient<IVrcaiMlaTest>(ipEndPoint))
                inputs = client.Proxy.getObs(new List<float>(vectorAction));

            //Debug.Log(observations.GetType());
            //for (int i = 0; i < observations.Count; i++)
            //{
            //    Debug.Log(observations[i].ToString());
            //}
        }
    }

    public override void AgentReset()
    {
        //gameObject.transform.rotation = new Quaternion(0f, 0f, 0f, 0f);
        //gameObject.transform.Rotate(new Vector3(1, 0, 0), Random.Range(-10f, 10f));
        //gameObject.transform.Rotate(new Vector3(0, 0, 1), Random.Range(-10f, 10f));
        //m_BallRb.velocity = new Vector3(0f, 0f, 0f);
        //ball.transform.position = new Vector3(Random.Range(-1.5f, 1.5f), 4f, Random.Range(-1.5f, 1.5f))
        //    + gameObject.transform.position;
        ////Reset the parameters when the Agent is reset.
        //SetResetParameters();
        Debug.Log("Episode done");
        if (stop_training == 0)
        {
            Debug.Log("Reseting agent");
            using (var client = new TcpClient<IVrcaiMlaTest>(ipEndPoint))
                client.Proxy.resetAgent();
        }
    }

    public override float[] Heuristic()
    {
        var action = new float[2];

        //action[0] = -Input.GetAxis("Horizontal");
        //action[1] = Input.GetAxis("Vertical");
        return action;
    }

    public void SetBall()
    {
        //Set the attributes of the ball by fetching the information from the academy
        //m_BallRb.mass = m_ResetParams.GetPropertyWithDefault("mass", 1.0f);
        //var scale = m_ResetParams.GetPropertyWithDefault("scale", 1.0f);
        //ball.transform.localScale = new Vector3(scale, scale, scale);
    }

    public void SetResetParameters()
    {
        //SetBall();
    }
}
