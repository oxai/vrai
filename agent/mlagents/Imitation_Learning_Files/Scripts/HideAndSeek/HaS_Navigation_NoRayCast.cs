using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;
using Barracuda;

// This script is for an environment with a single agent who's goal is 
// simply to navigate around all of the environment.
// The observations are the camera feed. 
// The actions are translations and rotations.
// If the agent learns to navigate the environment then
// we can use the trained model (nn) as a basis for more complex tasks
// such as playing games (Hide and Seek) etc.
// Note that hide and seek is essentially a navigation problem,
// hence the much simpler approach here.
// The aim is to use Imitation Learning (GAIL) + LSTM 
// to bootstrap the agents learning.

public class HaS_Navigation_NoRayCast : Agent{
    Rigidbody rb;
    public List<float> VectorActs = new List<float> {0,0,0}; // Actions output by NN
    float Speed = 10; // Current speed
    bool seekerLearning = true;
    HideAndSeekAcademy m_Academy;
    float seekerFloat;
    string behaviorName;

   // Start is called before the first frame update
    void Start(){

        behaviorName = GetComponent<BehaviorParameters>().behaviorName;
        m_Academy = FindObjectOfType<HideAndSeekAcademy>();
        rb = GetComponent<Rigidbody>();
    }


    public override void CollectObservations() {
	// The agent has camera feed, and No raycasts

//        for (int i = 0; i<10; i++) { // Ray cast for either wall or other agent 
//            float Distance = 1f;
//            RaycastHit hit;
//            if (Physics.Raycast(transform.position, transform.TransformDirection(PolarToCartesian(142, i * 36)), out hit, 142f)) {
//                Distance = hit.distance / 142f;
//                if (Application.isEditor) {
//                    Debug.DrawRay(transform.position, transform.TransformDirection(PolarToCartesian(hit.distance, i * 36)), Color.red, 0.01f, true);
//               }
//            }
//	AddVectorObs(Distance); // Distance to object 
//                }

    }




    public override void AgentReset() { // End of episode

        // Reset agent to its spawn point
            transform.eulerAngles = new Vector3(0, Random.Range(0f, 360f), 0);
            transform.position = transform.parent.gameObject.transform.position + new Vector3(Random.Range(-31, -23 ), 0, Random.Range(12, 15));
        // Reset action vector
        for (int i = 0; i<VectorActs.Count; i++) {
                VectorActs[i] = 0;
            }
    }



    public override void AgentAction(float[] vectorAction) {

        for (int i = 0; i<vectorAction.Length; i++) {
            VectorActs[i] = Mathf.Clamp(vectorAction[i],-1.0f,1.0f);
        }

        if (GetStepCount()>4000) { // Stop game after 1500 steps
            Done();}


      }




    private void FixedUpdate() {
        Vector3 new_velociy = new Vector3(0,0, VectorActs[1]);
        new_velociy = transform.TransformVector(new_velociy);
        rb.velocity = Vector3.Lerp(rb.velocity,Speed*new_velociy,Time.deltaTime*10);
        float new_angle = transform.eulerAngles.y -180f + VectorActs[2] * 180f;
        // Debug.Log(transform.eulerAngles.y);
        new_angle = (new_angle  > 180 ? new_angle - 360f : new_angle);
        new_angle = (new_angle  < -180 ? new_angle + 360f : new_angle);
        new_angle += 180f;
        transform.rotation = Quaternion.Lerp(transform.rotation,Quaternion.Euler(new Vector3(0, new_angle,0)),Time.deltaTime*5);
    }



            public override float[] Heuristic() {
                if (Input.GetKey(KeyCode.S))
                {
                    return new float[] { 0, -1f, 0 };
                }
                if (Input.GetKey(KeyCode.D))
                {
                    return new float[] { 1f, 0, 0 };
                }
                if (Input.GetKey(KeyCode.W))
                {
                    return new float[] { 0, 1f, 0 };
                }
                if (Input.GetKey(KeyCode.A))
                {
                    return new float[] { -1f, 0, 0 };
                }
                if (Input.GetKey(KeyCode.Q))
                {
                    return new float[] { 0, 0, -0.1f };
                }
                if (Input.GetKey(KeyCode.E))
                {
                    return new float[] { 0, 0, 0.1f };
                }
                return new float[] { 0, 0, 0 };
                // return VectorActs.ToArray();
            }

            public static Vector3 PolarToCartesian(float radius, float angle) {
                var x = radius * Mathf.Cos(angle * Mathf.PI / 180f);
                var z = radius * Mathf.Sin(angle * Mathf.PI / 180f);
                return new Vector3(x, 0f, z);
            }


    
}
