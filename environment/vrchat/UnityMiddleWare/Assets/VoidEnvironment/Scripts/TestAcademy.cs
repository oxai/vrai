using System;
using System.Linq;
using System.Text;
using ServiceWire;
using ServiceWire.TcpIp;
using System.Net;
using VrcaiMlaCommunicator;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using MLAgents;

public class TestAcademy : Academy
{
    public TcpClient<IVrcaiMlaTest> client;
    public IPEndPoint ipEndPoint;
    public override void InitializeAcademy()
    {
        //FloatProperties.RegisterCallback("gravity", f => { Physics.gravity = new Vector3(0, -f, 0); });
        string ip = "127.0.0.1";
        int port = 4420;
        ipEndPoint = new IPEndPoint(IPAddress.Parse(ip), port);
        //var zkEndpoint = new TcpZkEndPoint("username", "password",
        //    new IPEndPoint(IPAddress.Parse(ip), port), connectTimeOutMs: 2500);
        //client = new TcpClient<IVrcaiMlaTest>(ipEndPoint);

        //using (var client = new TcpClient<IVrcaiMlaTest>(zkEndpoint))
        //using (var )
        //{
            //List<float> actions = new List<float> { 0.1f, 0.1f, 0 };
            //List<float> response = client.Proxy.getObs(actions);
            //Debug.Log(response);
            //for (int i = 0; i < response.Count; i++)
            //{
            //    Debug.Log(response[i].ToString());
            //}
            //for (int i = 0; i < 10; i++)
            //{
            //    var id = client.Proxy.GetId("test1", 3.314, 42);
            //    int q2 = 4;
            //    var response = client.Proxy.Get(id, "mirror", 4.123, out q2);
            //    var list = client.Proxy.GetItems(id, new int[] { 3, 6, 9 });
            //}
        //}
    }

}
