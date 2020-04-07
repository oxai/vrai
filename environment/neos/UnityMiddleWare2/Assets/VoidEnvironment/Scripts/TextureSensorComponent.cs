using System;
using UnityEngine;

namespace MLAgents.Sensors
{
    public class TextureSensorComponent : SensorComponent
    {
        public string sensorName = "TextureSensor";
        public int width = 84;
        public int height = 84;
        public bool grayscale;
        public int num_channels = 3;
        TextureSensor sensor;

        public override ISensor CreateSensor()
        {
            sensor = new TextureSensor(height, width, grayscale, num_channels, sensorName);
            return sensor;
        }
        public void UpdateTexture(Texture2D texture)
        {
            sensor.UpdateTexture(texture);
        }

        public override int[] GetObservationShape()
        {
            //return new[] { height, width, grayscale ? 1 : 3 };
            return new[] { height, width, grayscale ? 1 : num_channels };
        }
    }
}
