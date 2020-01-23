using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using VRC;
using VRCModLoader;
using UnityEngine;


namespace AgentModule
{
    public class Agent
    {
        public List<float> VectorActs = new List<float> { 0, 0, 0 };
        public GameObject agentObject;
        public Transform transform;
        public Rigidbody rb;
        public float Speed = 10f;
        private List<Vector3> initialPoss;
        System.Random rnd = new System.Random();

        public Agent(GameObject agent)
        {
            agentObject = agent;
            transform = agent.transform;
            rb = agent.GetComponent<Rigidbody>();
            //rb.mass = 100f;
            rb.freezeRotation = true;
            initialPoss.Add(transform.position);
        }

        public void addInitPos()
        {
            initialPoss.Add(transform.position);
        }

        public void resetAgent()
        {
            Vector3 epsilon = new Vector3(UnityEngine.Random.Range(-0.1f, 0.1f), 0, UnityEngine.Random.Range(-0.1f, 0.1f));
            int index = rnd.Next(initialPoss.Count);
            transform.position = initialPoss[index] + epsilon;
        }

        public void updateActions(List<float> actions)
        {
            VectorActs = actions;
        }

        public List<float> getObservations(float stop_training)
        {
            Vector3 pos = transform.position;
            Quaternion rot = transform.rotation;
            Vector3 vel = rb.velocity;
            Vector3 rotVel = rb.angularVelocity;
            //NOTE: we are putting the reward as the last observation!
            return new List<float> { pos.x, pos.z, rot.x, rot.y, rot.z, rot.w, vel.x, vel.y, vel.z, rotVel.x, rotVel.y, rotVel.z, getReward(), stop_training };
        }

        public float getReward()
        {
            Vector3 pos = transform.position;
            Vector3 goal_pos = new Vector3(-10f, 0, -1.15f);
            return -(pos - goal_pos).magnitude;
        }

        public void move()
        {
            Vector3 new_velocity = new Vector3(VectorActs[0], 0, VectorActs[1]);
            new_velocity = transform.TransformVector(new_velocity);
            new_velocity[1] = 0;
            rb.AddForce(new_velocity*2f, ForceMode.VelocityChange);
            rb.velocity = Vector3.ClampMagnitude(rb.velocity, 3f);
            //rb.AddTorque(new Vector3(0, VectorActs[2]*10f, 0));
            //transform.position.y = Mathf.Clamp()
            //rb.velocity = Vector3.Lerp(rb.velocity, Speed * new_velocity, Time.deltaTime);
            float new_angle = transform.eulerAngles.y - 180f + VectorActs[2] * 10f;
            // Debug.Log(transform.eulerAngles.y);
            new_angle = (new_angle > 180 ? new_angle - 360f : new_angle);
            new_angle = (new_angle < -180 ? new_angle + 360f : new_angle);
            new_angle += 180f;
            transform.rotation = Quaternion.Lerp(transform.rotation, Quaternion.Euler(new Vector3(0, new_angle, 0)), Time.deltaTime);
        }
    }
}
