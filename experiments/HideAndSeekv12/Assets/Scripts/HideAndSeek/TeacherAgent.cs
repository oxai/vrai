using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class TeacherAgent : Agent{
    Rigidbody rb; 
    public List<float> VectorActs = new List<float> {0,0,0}; // Actions output by NN 
    public List<float> Inputs = new List<float>(); // Debugging variables to mointor inputs to NN I think
    public bool Seeker = true; // Determine whether agent is the seeker 
    bool Seeking = false; // Whether game is in seeking mode 
    float Speed = 100; // Current speed 
    bool hiderLearning = true; 
    bool seekerLearning = true;
    HideAndSeekAcademy m_Academy;
    // public HiderSeekerAgent Enemy; // Refernce to your enemy 
    float seekerFloat;
    bool sawEnemy = false;
    float waitPeriod = 240;
    string behaviorName;
    float distanceToSeenEnemey = 0.5f;
    float initial_return = 0;
    float final_return = 0;
    float student_return = 0;
    public StudentAgent Student;
    int episode_length = 700;

    // Start is called before the first frame update
    void Start(){
        behaviorName = GetComponent<BehaviorParameters>().behaviorName;
        m_Academy = FindObjectOfType<HideAndSeekAcademy>();
        if (Seeker) seekerFloat = 1f;
        else seekerFloat = 0f;
        seekerFloat = m_Academy.FloatProperties.GetPropertyWithDefault(behaviorName+"_seeking",seekerFloat);
        if (seekerFloat == 1f) Seeker = true;
        else if (seekerFloat == 0f) Seeker = false;
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
        // AddVectorObs(transform.localPosition.x / 50f);
        // AddVectorObs(transform.localPosition.z / 50f);
        AddVectorObs((transform.eulerAngles.y > 180 ? transform.eulerAngles.y - 360f : transform.eulerAngles.y)/180f);
        for (int i = 0; i<5; i++) { // Ray cast for either wall or other agent 
            float Wall = 0;
            float HiderOrSeeker = 0;
            float Distance = 1f;
            float seenThingAngle = 0;
            float seenThingVelocity_x = 0;
            float seenThingVelocity_y = 0;
            RaycastHit hit;
            if (Physics.Raycast(transform.position, transform.TransformDirection(PolarToCartesian(142, 70 + i * 10)), out hit, 142f)) {
                seenThingAngle=(hit.transform.eulerAngles.y > 180 ? hit.transform.eulerAngles.y - 360f : hit.transform.eulerAngles.y)/180f;
                if (hit.rigidbody){
                    seenThingVelocity_x = hit.rigidbody.velocity.x/10f;
                    seenThingVelocity_y = hit.rigidbody.velocity.y/10f;
                }
                Distance = hit.distance / 142f;
                if (hit.transform.tag == "agent") {
                    // Debug.Log("Saw enemy");
                    HiderOrSeeker = 1;
                    sawEnemy = true;
                    distanceToSeenEnemey = Distance;
                } else {
                    Wall = 1;
                }
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
            AddVectorObs(seenThingAngle);
            AddVectorObs(seenThingVelocity_x);
            AddVectorObs(seenThingVelocity_y);
        }
    }

    public override void AgentReset() { // End of episode 
        sawEnemy = false;
        // seekerFloat = m_Academy.FloatProperties.GetPropertyWithDefault(behaviorName+"_seeking",seekerFloat);
        // print(seekerFloat);
        // waitPeriod = m_Academy.FloatProperties.GetPropertyWithDefault("waitPeriod",240);
        // if (seekerFloat == 1f) Seeker = true;
        // else if (seekerFloat == 0f) Seeker = false;
        // distanceToSeenEnemey = 0.5f;

        // Seeking = false;
        // // Reset agent to its spawn point
        // if (behaviorName == "seeker") {
        //     // transform.localPosition = new Vector3(Random.Range(0f,14f),0,Random.Range(11f,-11f));
        //     // transform.eulerAngles = new Vector3(0,Random.Range(0f,360f),0);
        //     transform.eulerAngles = new Vector3(0, Random.Range(0f, 360f), 0);
        //     transform.position = transform.parent.gameObject.transform.position + new Vector3(Random.Range(10f, -10f), 0, Random.Range(10f, -10f));
        // } else {
        //     transform.eulerAngles = new Vector3(0, Random.Range(0f, 360f), 0);
        //     transform.position = transform.parent.gameObject.transform.position + new Vector3(Random.Range(10f, -10f), 0, Random.Range(10f, -10f));
        // }
        // Reset action vector
        // for (int i = 0; i<VectorActs.Count; i++) {
        //         VectorActs[i] = 0;
        //     }
        // VectorActs = new List<float>{transform.position.x/20f, 0, transform.position.z/20f};
    }

    public override void AgentAction(float[] vectorAction) {
        int step = GetStepCount();
        // Debug.Log(step);
        if (step == 0) {
            for (int i = 0; i<vectorAction.Length; i++) {
                VectorActs[i] = Mathf.Clamp(vectorAction[i],-1f,1f);
            }
        }
        // if (step>episode_length) {
        //     // for (int i = 0; i<VectorActs.Count; i++) {
        //             // VectorActs[i] = new Vector3(transform.position.x/20f, 0, transform.position.z/20f); 
        //     // VectorActs = new List<float>{transform.position.x/20f, 0, transform.position.z/20f};
        //         // }
        // } else { //in new approach without observations, I should be acting on first teacher step, but welp
        //     for (int i = 0; i<vectorAction.Length; i++) {
        //         VectorActs[i] = Mathf.Clamp(vectorAction[i],-1f,1f);
        //     }
        //     // Debug.Log(VectorArts[0]);
        // }
        Debug.Log(student_return);
        if (step%episode_length == 0) {
            student_return = 0;
        }
        // if (step <= episode_length*4) {
        //     initial_return += student_return;
        // }
        // if (step == episode_length*4) {
        //     initial_return /= 4;
        // }
        if (step > episode_length*4 && step <= episode_length*8) {
            AddReward(Mathf.Abs(student_return-initial_return)/4);
        }
        if (step>=episode_length*8) {
            Done();
        }
    }

    public void updateReturn(float reward) {
        student_return += reward;
    }

    private void FixedUpdate() {
        // Debug.Log(VectorActs[0]);
        transform.position = transform.parent.gameObject.transform.position + new Vector3(40*VectorActs[0], 0, 40*VectorActs[2]);
        // if (Seeker && !Seeking) // Do nothing when not seeking as the seeker 
        //     return;
        // Convert NN output to action 
        // Lerp is to interpolate between the current velocity (rb.velocity)
        // and the target velocity
        // in a time Time.deltaTime*10
        // we use that because we take a decision every 10 frames.
        // The actions have interpretation of (speed, angle of velocity in x-z plane, and angle)
        // rb.velocity = Vector3.Lerp(rb.velocity,Speed*VectorActs[0]*new Vector3(Mathf.Sin(VectorActs[1]*Mathf.PI),0, Mathf.Cos(VectorActs[1] * Mathf.PI)),Time.deltaTime*10);
        // Vector3 new_velociy = new Vector3(VectorActs[0],0, VectorActs[1]);
        // new_velociy = transform.TransformVector(new_velociy);
        // rb.velocity = Vector3.Lerp(rb.velocity,Speed*new_velociy,Time.deltaTime*10);
        // float new_angle = transform.eulerAngles.y -180f + VectorActs[2] * 180f;
        // // Debug.Log(transform.eulerAngles.y);
        // new_angle = (new_angle  > 180 ? new_angle - 360f : new_angle);
        // new_angle = (new_angle  < -180 ? new_angle + 360f : new_angle);
        // new_angle += 180f;
        // transform.rotation = Quaternion.Lerp(transform.rotation,Quaternion.Euler(new Vector3(0, new_angle,0)),Time.deltaTime*5);
        // transform.rotation = Quaternion.Euler(new Vector3(0, new_angle,0));
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

    // private void OnCollisionEnter(Collision collision) { // Rewards for being caught
    //     if (Seeking&&collision.transform.CompareTag("agent")) {
    //         // if (!Seeker) AddReward(-10f);
    //         // else AddReward(10f);
    //         // Student.AddReward(10f);
    //         // student_return += 10f; 
    //         // Student.Done();
    //     }

    // }
}
