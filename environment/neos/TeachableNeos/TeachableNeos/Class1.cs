using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Grpc.Core;

namespace TeachableNeos
{
    class DataCommImpl: DataComm.DataCommBase
    {
        public DataCommImpl()
        {

        }

        public override Task<Classification> SendFeatures(Feature f, ServerCallContext context)
        {
            return Task.FromResult(new Classification
            {
                K = 2
            });
        }
    }
}
