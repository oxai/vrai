using System;
using System.Collections;
using UnityEngine;
using VRC;
using VRCMenuUtils;
using VRCModLoader;

namespace ExampleModule
{

    [VRCModInfo("VRCAI", "0.2.5", "BeansMain", null, null)]
    public class VRCAI : VRCMod
	{
		
		public void OnGUI()
		{

		}
        public float Speed = 0.7f;
        //public Renderer rend;
        //private void OnApplicationStart()
        //{
        //    ModManager.StartCoroutine(this.Setup());
        //}
        //private IEnumerator Setup()
        //{
        //    yield return VRCMenuUtilsAPI.WaitForInit();
        //    VRCMenuUtilsAPI.AddQuickMenuButton("Rgb", "Rainbow\nRGB\nOn", "Enables/Disables Rainbow nameplates", new Action(this.RGB_On));
        //    VRCMenuUtilsAPI.AddQuickMenuButton("Rgb", "Rainbow\nRGB\nOff", "Enables/Disables Rainbow nameplates", new Action(this.RGB_Off));
        //    yield break;
        //}
        public void Start()
		{
            Console.WriteLine("[VRCAI] Started.");
		}

		
		public void FixedUpdate()
		{
            this.GetPlayerPositions();
		}
        public void GetPlayerPositions()
		{
			foreach (Player player in PlayerManager.GetAllPlayers())
			{
                String name = player.vrcPlayer.namePlate.mainText.text; //not sure if this is the best way to get username
                Vector3 position = player.vrcPlayer.avatarGameObject.transform.position;
                Transform chest = player.vrcPlayer.avatarGameObject.transform.Find("Armature").Find("Hips").Find("Spine").Find("Chest");
                Transform right_hand = chest.Find("Right_shoulder").Find("Right_arm").Find("Right_elbow").Find("Right_wrist");
                Vector3 right_hand_position = right_hand.position;
                Transform left_hand = chest.Find("Left_shoulder").Find("Left_arm").Find("Left_elbow").Find("Left_wrist");
                Vector3 left_hand_position = left_hand.position;
                using (System.IO.StreamWriter file = new System.IO.StreamWriter("player_"+name+"_data.txt", true))
                {
                    file.WriteLine(position.ToString("F6") + "\t" + left_hand_position.ToString("F6") + "\t" + right_hand_position.ToString("F6"));
                }
            }
		}

        //public static bool toggle;
	}
}
