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

        public override Task<NeosObservation> GetObs(Empty f, ServerCallContext context)
        {
            return Task.FromResult(new NeosObservation
            {
                X = 1, Y = 1, Z = 1
            });
        }
    }
}
