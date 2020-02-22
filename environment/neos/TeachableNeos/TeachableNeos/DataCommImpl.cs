using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Grpc.Core;
using FrooxEngine.LogiX.Network;

namespace TeachableNeos
{
    class DataCommImpl: DataComm.DataCommBase
    {
        public ServeData node;
        public DataCommImpl(ServeData parent_node)
        {
            node = parent_node;
        }

        public override Task<NeosObservation> GetObs(Empty f, ServerCallContext context)
        {
            return Task.FromResult(new NeosObservation
            {
                X = node.x.EvaluateRaw(), Y = 1, Z = 1
            });
        }
    }
}
