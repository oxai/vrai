using ServiceWire;
using ServiceWire.TcpIp;
//using ServiceWire.ZeroKnowledge;
using System;
using System.Net;
using System.Collections;
using System.Linq;
using System.Text;
using UnityEngine;
using VRC;
using VRCMenuUtils;
using VRCModLoader;
using VrcaiMlaCommunicator;
using System.Collections.Generic;
using AgentModule;
using VRCAI;
using VRCTools;
//using System.Threading.Tasks;
//using System.Net;
//using Windows.Networking.Sockets;
//using System.Net.sockets;
using System.Reflection;
using HumanData;


namespace vrcai
{

    [VRCModInfo("VRCAI", "0.2.5", "oxai", null, null)]
    public class VRCAI : VRCMod
	{
        public static Agent agent;
        public static GameObject stand;
        public static List<GameObject> joints;
        public static Player myPlayer;
        public static bool wiggly_avatar = false;
        public static Vector3 avatarPos1;
        public static Vector3 avatarPos2;
        public static int avatarState = 1;
        public static int timeScaleState = 1;
        public static float stop_training = 0;
        private bool recording = false;
        public void OnGUI()
		{

		}
        //private IEnumerator Setup()
        //{
        //    yield return VRCMenuUtilsAPI.WaitForInit();
        //    VRCMenuUtilsAPI.AddQuickMenuButton("Rgb", "Rainbow\nRGB\nOn", "Enables/Disables Rainbow nameplates", new Action(this.RGB_On));
        //    VRCMenuUtilsAPI.AddQuickMenuButton("Rgb", "Rainbow\nRGB\nOff", "Enables/Disables Rainbow nameplates", new Action(this.RGB_Off));
        //    yield break;
        //}
        private void OnApplicationStart()
        {
            ModManager.StartCoroutine(StartRPCServer());
        }
        void OnLevelWasloaded(int level)
        {
            if (level != -1) return;
            //ModManager.StartCoroutine(FindShiba());
            System.Reflection.PropertyInfo propertyInfo;
        }
        public void Update()
        {
            if (Input.GetKeyDown(KeyCode.F9))

            {
                GameObject shiba_big = GameObject.Find("prop_shibaplush (1)");
                if (!(shiba_big == null)) agent = new Agent(shiba_big);
                //stand = GameObject.Find("avatar_stand");
                stand = GameObject.Find("vrcai_ava_armature");
                VRCModLoader.VRCModLogger.Log(stand.name);
                joints = utils.findAllChildren(stand);
                VRCModLoader.VRCModLogger.Log(joints[0].transform.rotation.ToString());
            }
            if (Input.GetKeyDown(KeyCode.F3))
            {
                myPlayer = PlayerManager.GetAllPlayers()[0];
                wiggly_avatar = !wiggly_avatar;
                avatarPos1 = myPlayer.vrcPlayer.transform.position;
                avatarPos2 = avatarPos1 + new Vector3(0.1f, 0.1f, 0.1f);
            }
            if (Input.GetKeyDown(KeyCode.T))
            {
                if (stop_training == 0)
                {
                    stop_training = 1f;
                    Time.timeScale = 1f;
                }
                else if (stop_training == 1f)
                {
                    stop_training = 0;
                    Time.timeScale = 20f;
                }
            }
            if (Input.GetKeyDown(KeyCode.P))
            {
                agent.addInitPos();
            }
            if (Input.GetKeyDown(KeyCode.R))
            {
                if (!recording)
                {
                    recording = true;
                }
                else
                {
                    recording = false;
                }
            }
        }
                internal IEnumerator FindShiba()
        {
            while (RoomManager.currentRoom == null)
            {
                yield return new WaitForSeconds(1f);
            }
            GameObject shiba_big = GameObject.Find("prop_shibaplush (1)");
            if (!(shiba_big == null))
            {
                agent = new Agent(shiba_big);
            }
            else
            {
                agent = null;
            }
        }
        internal IEnumerator StartRPCServer()
        {
            int port = 4420;
            var ipEndpoint = new IPEndPoint(IPAddress.Any, port);

            var vrcaimltest = new VrcaiMlaTest();

            var tcphost = new TcpHost(ipEndpoint);
            tcphost.AddService<IVrcaiMlaTest>(vrcaimltest);

            VRCModLoader.VRCModLogger.Log("Starting server");
            tcphost.Open();
            while (true) { yield return null; }
            tcphost.Close();
        }

        public void FixedUpdate()
		{

            if (!(agent == null))
            {
                VRCModLoader.VRCModLogger.Log("Agent moving");
                agent.move();
                agent.texture.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0, false);
            }

            if (recording)
            {
                PlayerRecorder.recordPlayersVariables();
            }

            if (wiggly_avatar)
            {
                if (avatarState == 1)
                {
                    myPlayer.vrcPlayer.transform.position = avatarPos1;
                    avatarState = 2;
                } else if (avatarState == 2)
                {
                    myPlayer.vrcPlayer.transform.position = avatarPos2;
                    avatarState = 1;
                }
            }
        }

        private void OnPostRender()
        {
            VRCModLoader.VRCModLogger.Log("HIIIIIIIIIIIIIIII");
            
        }

        //public static List<float> getObs(List<float> actions)
        public static TextureMessage getObs(List<float> actions)
        {
            agent.updateActions(actions);
            return agent.getObservations(stop_training);
        }

        //public static bool toggle;
	}

    public class VrcaiMlaTest : IVrcaiMlaTest
    {
        //public List<float> getObs(List<float> actions)
        public TextureMessage getObs(List<float> actions)
        {
            //float observation = 1f;
            //return Task.FromResult(observation);
            //return 1f;
            return VRCAI.getObs(actions);
        }
        public void resetAgent()
        {
            //float observation = 1f;
            //return Task.FromResult(observation);
            //return 1f;
            VRCAI.agent.resetAgent();
        }
    }
}
