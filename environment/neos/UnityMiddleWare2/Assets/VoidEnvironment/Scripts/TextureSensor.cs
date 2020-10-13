using System;
using UnityEngine;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Sensors
{
    public class TextureSensor : ISensor
    {
        Texture2D m_Texture;
        bool m_Grayscale;
        string m_Name;
        int[] m_Shape;

        public TextureSensor(int height, int width, bool grayscale, int num_channels, string name)
        {
            //var width = m_Texture != null ? m_Texture.width : 0;
            //var height = m_Texture != null ? m_Texture.height : 0;
            m_Grayscale = grayscale;
            m_Name = name;
            m_Shape = new[] { height, width, grayscale ? 1 : num_channels };
        }

        public string GetName()
        {
            return m_Name;
        }

        public int[] GetObservationShape()
        {
            return m_Shape;
        }

        public byte[] GetCompressedObservation()
        {
            var texture = m_Texture;
            // TODO support more types here, e.g. JPG
            var compressed = texture.EncodeToPNG();
            UnityEngine.Object.Destroy(texture);
            return compressed;
        }

        public int Write(ObservationWriter writer)
        {
            var texture = m_Texture;
            var numWritten = writer.WriteTexture(texture, m_Grayscale);
            UnityEngine.Object.Destroy(texture);
            return numWritten;
        }

        public void Update() { }

        public SensorCompressionType GetCompressionType()
        {
            return SensorCompressionType.PNG;
        }
        public void UpdateTexture(Texture2D texture)
        {
            m_Texture = texture; 
        }
        public void Reset() { }
    }
}
