using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;

namespace VRCAI
{
    class utils
    {
        public static List<GameObject> findAllChildren(GameObject root)
        {
            List<GameObject> descendants = new List<GameObject>();
            Transform t = root.transform;
            for (int i = 0; i < t.childCount; i++)
            {
                GameObject child = t.GetChild(i).gameObject;
                descendants.Add(child);
                if (child.transform.childCount > 0)
                {
                    descendants.AddRange(findAllChildren(child));
                }
            }

            return descendants;
        }

        //public static List<Quaternion> 
    }
}
