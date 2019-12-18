using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class HiderSeekerAgent : Agent{
    Rigidbody rb; 
    public List<float> VectorActs = new List<float> {0,0,0}; // Actions output by NN 
    public List<float> Inputs = new List<float>(); // Debugging variables to mointor inputs to NN I think
    public bool Seeker = true; // Determine whether agent is the seeker 
    bool Seeking = false; // Whether game is in seeking mode 
    float Speed = 10; // Current speed 
    bool hiderLearning = true; 
    bool seekerLearning = true;
    HideAndSeekAcademy m_Academy;
    float seekerFloat;

    // Start is called before the first frame update
    void Start(){
        m_Academy = FindObjectOfType<HideAndSeekAcademy>();
        if (Seeker) seekerFloat = 1f;
        else seekerFloat = 0f;
        seekerFloat = m_Academy.FloatProperties.GetPropertyWithDefault("seeker",seekerFloat);
        if (seekerFloat == 1f) Seeker = true;
        else if (seekerFloat == 0f) Seeker = false;
        if (Seeker) {// Seeker moves faster? 
            Speed = 12;
        } else {
            Speed = 10;
            // Sometimes one agent learns, sometimes the other learns. Hmm, I guess this could help give either hider and seekers, an advantage
            // if (Random.value < 0.5) {
            //     hiderLearning = false;
            //     seekerLearning = true;
            // } else {
            //     hiderLearning = true;
            //     seekerLearning = false;
            // }
        }
        rb = GetComponent<Rigidbody>(); 
        for (int i = 0; i<20; i++) { // Set all inputs to zero 
            Inputs.Add(0);
        }
    }

    public override void CollectObservations() {
        // Debugging inputs
        //Inputs[0] = rb.velocity.x / 10f;
        //Inputs[1] = rb.velocity.z / 10f;
        //Inputs[2] = transform.localPosition.x / 50f;
        //Inputs[3] = transform.localPosition.z / 50f;
        //Inputs[4] = (transform.eulerAngles.y > 180 ? transform.eulerAngles.y - 360f : transform.eulerAngles.y)/180f;
        // float seekerIndicator = 0;
        // if (Seeker) seekerIndicator = 1f;
        // If seeker we add 0 elements to first half
        // AddVectorObs(seekerIndicator); //Input to indicate to neural whether this is hider or seeker
        if (!Seeking&& GetStepCount()<240) { // While not in seeking mode, set 0th index to count-down  
            Inputs[0] = (240 - GetStepCount()) / 240f;
            AddVectorObs((240 - GetStepCount()) / 240f);
        } else {
            AddVectorObs(0); // While in seeking mode, set 0th index to 0 
            Inputs[0] = 0;
        }
        AddVectorObs(rb.velocity.x / 10f); // Transform variables 
        AddVectorObs(rb.velocity.z / 10f);
        AddVectorObs(transform.localPosition.x / 50f);
        AddVectorObs(transform.localPosition.z / 50f);
        AddVectorObs((transform.eulerAngles.y > 180 ? transform.eulerAngles.y - 360f : transform.eulerAngles.y)/180f);
        for (int i = 0; i<5; i++) { // Ray cast for either wall or other agent 
            float Wall = 0;
            float HiderOrSeeker = 0;
            float Distance = 1f;
            RaycastHit hit;
            if (Physics.Raycast(transform.position, transform.TransformDirection(PolarToCartesian(142, 70 + i * 10)), out hit, 142f)) {
                if (hit.transform.tag == "agent") {
                    HiderOrSeeker = 1;
                } else {
                    Wall = 1;
                }
                Distance = hit.distance / 142f;
                if (Application.isEditor) {
                    Debug.DrawRay(transform.position,
                        transform.TransformDirection(
                        PolarToCartesian(hit.distance, 70 + i * 10)), Color.black, 0.01f, true);
                }
            } else {
                if (Application.isEditor) {
                    Debug.DrawRay(transform.position,
                        transform.TransformDirection(
                        PolarToCartesian(142, 70 + i * 10)), Color.black, 0.01f, true);
                }
            }
            //Inputs[5 + i * 3] = Wall;
            //Inputs[6 + i * 3] = HiderOrSeeker;
            //Inputs[7 + i * 3] = Distance;
            AddVectorObs(Wall);  // If wall found 
            AddVectorObs(HiderOrSeeker); // If hider or seeker found 
            AddVectorObs(Distance); // Distance to object
        }
    }

    public override void AgentReset() { // End of episode 

        Seeking = false;
        // Reset agent to its spawn point
        if (Seeker) {
            transform.localPosition = new Vector3(Random.Range(0f,14f),0,Random.Range(11f,-11f));
            transform.eulerAngles = new Vector3(0,Random.Range(0f,360f),0);
        } else {
            transform.eulerAngles = new Vector3(0, Random.Range(0f, 360f), 0);
            transform.localPosition = new Vector3(Random.Range(-14f, 0f), 0, Random.Range(11f, -11f));
        }
        // Reset action vector
        for (int i = 0; i<VectorActs.Count; i++) {
                VectorActs[i] = 0;
            }
    }

    public override void AgentAction(float[] vectorAction) {
        if (Seeking) { // In seeking mode 
            if (Seeker && seekerLearning) { 
                AddReward(-0.002f); // Negative survival 
            } else if (hiderLearning) {
                AddReward(0.002f); // Positive survival 
            }
        } else {
            if (GetStepCount() >= 240) {   // Wait until seeking mode 
                Seeking = true;
            }
        }
        if (Seeker && !Seeking) // Do nothing when not seeking as the seeker
            return;
        for (int i = 0; i<vectorAction.Length/2; i++) {
            VectorActs[i] = Mathf.Clamp(vectorAction[i],-1.0f,1.0f);
        }
        
        // if (Seeker) {
        //     for (int i = 0; i<vectorAction.Length/2; i++) {
        //         VectorActs[i] = Mathf.Clamp(vectorAction[i],-1.0f,1.0f);
        //     }
        // } else {
        //     for (int i = vectorAction.Length/2; i<vectorAction.Length; i++) {
        //         VectorActs[i-vectorAction.Length/2] = Mathf.Clamp(vectorAction[i],-1.0f,1.0f);
        //     }
        // }
        
        if (GetStepCount()>1500) { // Stop game after 1500 steps 
            if (Seeker && seekerLearning) AddReward(-1f);
            else if (hiderLearning) AddReward(1f);
            Done();
        }
    }

    private void FixedUpdate() {
        if (Seeker && !Seeking) // Do nothing when not seeking as the seeker 
            return;
        // Convert NN output to action 
        // Lerp is to interpolate between the current velocity (rb.velocity)
        // and the target velocity
        // in a time Time.deltaTime*10
        // we use that because we take a decision every 10 frames.
        // The actions have interpretation of (speed, angle of velocity in x-z plane, and angle)
        rb.velocity = Vector3.Lerp(rb.velocity,Speed*VectorActs[0]*new Vector3(Mathf.Sin(VectorActs[1]*Mathf.PI),0, Mathf.Cos(VectorActs[1] * Mathf.PI)),Time.deltaTime*10);
        transform.rotation = Quaternion.Lerp(transform.rotation,Quaternion.Euler(new Vector3(0, VectorActs[2] * 180f,0)),Time.deltaTime*5);
    }

    public override float[] Heuristic() {
        return VectorActs.ToArray();
    }

    public static Vector3 PolarToCartesian(float radius, float angle) {
        var x = radius * Mathf.Cos(angle * Mathf.PI / 180f);
        var z = radius * Mathf.Sin(angle * Mathf.PI / 180f);
        return new Vector3(x, 0f, z);
    }

    private void OnCollisionEnter(Collision collision) { // Rewards for being caught
        if (Seeking&&collision.transform.CompareTag("agent")) {
            if (!Seeker) AddReward(-1f);
            else AddReward(1f);
            Done();
        }

    }
}
