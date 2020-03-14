using System;
using UnityEngine;

namespace MLAgents.Sensor
{
    public class CameraSensorComponent : SensorComponent
    {
        public new Camera camera;
        public string sensorName = "CameraSensor";
        public int width = 84;
        public int height = 84;
        public bool grayscale;
        public int num_channels = 3;

        public override ISensor CreateSensor()
        {
            return new CameraSensor(camera, width, height, grayscale, sensorName);
        }

        public override int[] GetObservationShape()
        {
            //return new[] { height, width, grayscale ? 1 : 3 };
            return new[] { height, width, grayscale ? 1 : num_channels };
        }
    }
}
