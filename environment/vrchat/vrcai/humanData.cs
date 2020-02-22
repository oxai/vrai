using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;
using VRC;
using VRCMenuUtils;
using VRCModLoader;

namespace HumanData
{
    class PlayerRecorder
    {
        public static void recordPlayersVariables()
        {
            foreach (Player player in PlayerManager.GetAllPlayers())
            {
                String name = player.vrcPlayer.namePlate.mainText.text; //not sure if this is the best way to get username
                //Vector3 position = player.vrcPlayer.avatarGameObject.transform.position;
                Vector3 position = player.gameObject.transform.position;
                Vector3 rotation = player.gameObject.transform.rotation;
                //Transform chest = player.vrcPlayer.avatarGameObject.transform.Find("Armature").Find("Hips").Find("Spine").Find("Chest");
                //Transform right_hand = chest.Find("Right_shoulder").Find("Right_arm").Find("Right_elbow").Find("Right_wrist");
                //Vector3 right_hand_position = right_hand.position;
                //Transform left_hand = chest.Find("Left_shoulder").Find("Left_arm").Find("Left_elbow").Find("Left_wrist");
                //Vector3 left_hand_position = left_hand.position;
                Transform ik = player.gameObject.transform.Find("AnimationController").Find("HeadAndHandIK");
                Transform HeadEffector = ik.Find("HeadEffector");
                Transform LeftEffector = ik.Find("LeftEffector");
                Transform RightEffector = ik.Find("RightEffector");
                Vector3 head_rotation = HeadEffector.rotation.eulerAngles;
                Vector3 left_hand_position = LeftEffector.position;
                Vector3 left_hand_rotation = LeftEffector.rotation.eulerAngles;
                Vector3 right_hand_position = RightEffector.position;
                Vector3 right_hand_rotation = RightEffector.rotation.eulerAngles;
                List<Vector3> variables = new List<Vector3>
                    {position, rotation, head_rotation, left_hand_position, left_hand_rotation, right_hand_position, right_hand_rotation};
                using (System.IO.StreamWriter file = new System.IO.StreamWriter("player_" + name + "_data.txt", true))
                {
                    file.WriteLine(String.Join("\t",(from vec in variables select vec.ToString("F6")).ToArray()));
                }

            }
        }
    }
}
