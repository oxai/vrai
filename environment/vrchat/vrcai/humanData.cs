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
        public void recordPlayersPositions()
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
                using (System.IO.StreamWriter file = new System.IO.StreamWriter("player_" + name + "_data.txt", true))
                {
                    file.WriteLine(position.ToString("F6") + "\t" + left_hand_position.ToString("F6") + "\t" + right_hand_position.ToString("F6"));
                }
            }
        }
    }
}
