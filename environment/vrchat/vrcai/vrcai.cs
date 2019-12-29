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

		
		public void Update()
		{
            this.GetPlayerPositions();
		}
        public void GetPlayerPositions()
		{
			foreach (Player player in PlayerManager.GetAllPlayers())
			{
                Vector3 positions = player.vrcPlayer.avatarGameObject.transform.position;
                using (System.IO.StreamWriter file =
                new System.IO.StreamWriter(@"player_positions.txt", true))
                    {
                        file.WriteLine(positions.ToString());
                    }
            }
		}

        //public static bool toggle;
	}
}
