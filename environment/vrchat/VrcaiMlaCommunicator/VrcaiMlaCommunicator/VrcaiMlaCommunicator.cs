using System;
using System.Collections.Generic;
//using UnityEngine;
//using System.Threading.Tasks;

namespace VrcaiMlaCommunicator
{
    public interface IVrcaiMlaTest
    {
        //Task<float> GetObs(float action);
        //List<float> getObs(List<float> actions);
        TextureMessage getObs(List<float> actions);
        void resetAgent();
    }
    [Serializable]
    public struct TextureMessage
    {
        public byte[] raw_texture { get; set; }
        public int size { get; set; }
        public int width { get; set; }
        public int height { get; set; }
    }
}
