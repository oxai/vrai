using System;
using System.Collections.Generic;
//using System.Threading.Tasks;

namespace VrcaiMlaCommunicator
{
    public interface IVrcaiMlaTest
    {
        //Task<float> GetObs(float action);
        List<float> getObs(List<float> actions);
        void resetAgent();
    }
}
