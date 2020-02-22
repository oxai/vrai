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
using System.Threading;
using Microsoft.PerceptionSimulation;
using Vector3 = UnityEngine.Vector3;

public class TestAcademy : Academy
{
    public TcpClient<IVrcaiMlaTest> client;

    public IPEndPoint ipEndPoint;

    private RestSimulationStreamSink sink;

    private ISimulatedHuman2 human;
    private IPerceptionSimulationManager manager;

    private CancellationToken cancellationToken;
    //public do_action;
    //public Dictionary<string, Vector3> actions;
    public override async void InitializeAcademy()
    {
        //FloatProperties.RegisterCallback("gravity", f => { Physics.gravity = new Vector3(0, -f, 0); });
        //do_action = false;
        string ip = "127.0.0.1";
        int port = 4420;
        ipEndPoint = new IPEndPoint(IPAddress.Parse(ip), port);
        sink = null;

        cancellationToken = new System.Threading.CancellationToken();

        try
        {
            sink = await RestSimulationStreamSink.Create(
                new Uri("http://" + ip + ":50080/"),
                new System.Net.NetworkCredential("", ""),
                true
            );

            manager = PerceptionSimulationManager.CreatePerceptionSimulationManager(sink);
            Console.WriteLine("hi");
            //human = (ISimulatedHuman2) manager.Human;
            //human.LeftController.Status = SimulatedSixDofControllerStatus.Active;
            //human.RightController.Status = SimulatedSixDofControllerStatus.Active;
        }
        catch (Exception e)
        {
            Console.WriteLine(e);
            Debug.Log(e);
        }

    }

    public void DoAction(Dictionary<string, Vector3> actions)
    {
        //Debug.Log(actions["move"]);
        //Debug.Log(human);
        foreach (KeyValuePair<string, Vector3> entry in actions)
        {
            string action_type = entry.Key;
            Microsoft.PerceptionSimulation.Vector3 value = new Microsoft.PerceptionSimulation.Vector3(entry.Value[0],entry.Value[1],entry.Value[2]);
            if (action_type == "move")
            {
                human.Move(value);
            }
            else if (action_type == "rotate")
            {
                float yaw_rot = value.Y;
                human.Rotate(yaw_rot);
            }
            else if (action_type == "rotate_head")
            {
                Microsoft.PerceptionSimulation.Rotation3 rot = new Microsoft.PerceptionSimulation.Rotation3(value.X,value.Y, value.Z);
                human.Head.Rotate(rot);
            }
            else if (action_type == "move_left_hand")
            {
                human.LeftController.Move(value);
            }
            else if (action_type == "rotate_left_hand")
            {
                var or = human.LeftController.Orientation;
                Quaternion rot = Quaternion.Euler(new Vector3(or.Pitch,or.Yaw,or.Roll));
                Quaternion deltaRot = Quaternion.Euler(new Vector3(value.X,value.Y,value.Z));
                Vector3 newAngle = (rot * deltaRot).eulerAngles;
                human.LeftController.Orientation = new Rotation3(newAngle.x,newAngle.y,newAngle.z);
            }
            else if (action_type == "move_left_hand")
            {
                human.RightController.Move(value);
            }
            else if (action_type == "rotate_left_hand")
            {
                var or = human.RightController.Orientation;
                Quaternion rot = Quaternion.Euler(new Vector3(or.Pitch,or.Yaw,or.Roll));
                Quaternion deltaRot = Quaternion.Euler(new Vector3(value.X,value.Y,value.Z));
                Vector3 newAngle = (rot * deltaRot).eulerAngles;
                human.RightController.Orientation = new Rotation3(newAngle.x,newAngle.y,newAngle.z);
            }
        }
    }

    protected override async void OnDestroy()
    {
        base.OnDestroy();
        if (sink != null)
        {
            await sink.Close(cancellationToken);
        }
    }

}
