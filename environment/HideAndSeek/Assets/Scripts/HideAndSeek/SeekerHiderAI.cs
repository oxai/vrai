using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class SeekerHiderAI : Agent{
    Rigidbody rb;
    public List<float> VectorActs = new List<float> {0,0,0};
    public List<float> Inputs = new List<float>();
    public bool Seeker = true;
    bool Seeking = false;
    public SeekerHiderAI Enemy;
    float Speed = 10;

    // Start is called before the first frame update
    void Start(){
        if (Seeker)
            Speed = 12;
        rb = GetComponent<Rigidbody>();
        for (int i = 0; i<20; i++) {
            Inputs.Add(0);
        }
    }

    public override void CollectObservations() {
        //Inputs[0] = rb.velocity.x / 10f;
        //Inputs[1] = rb.velocity.z / 10f;
        //Inputs[2] = transform.localPosition.x / 50f;
        //Inputs[3] = transform.localPosition.z / 50f;
        //Inputs[4] = (transform.eulerAngles.y > 180 ? transform.eulerAngles.y - 360f : transform.eulerAngles.y)/180f;
        if (!Seeking&& GetStepCount()<240) {
            Inputs[0] = (240 - GetStepCount()) / 240f;
            AddVectorObs((240 - GetStepCount()) / 240f);
        } else {
            AddVectorObs(0);
            Inputs[0] = 0;
        }
        AddVectorObs(rb.velocity.x / 10f);
        AddVectorObs(rb.velocity.z / 10f);
        AddVectorObs(transform.localPosition.x / 50f);
        AddVectorObs(transform.localPosition.z / 50f);
        AddVectorObs((transform.eulerAngles.y > 180 ? transform.eulerAngles.y - 360f : transform.eulerAngles.y)/180f);
        for (int i = 0; i<5; i++) {
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
            AddVectorObs(Wall);
            AddVectorObs(HiderOrSeeker);
            AddVectorObs(Distance);
        }
    }

    public override void AgentReset() {
        if (!Seeker) {
            Seeking = false;
            transform.localPosition = new Vector3(Random.Range(0f,14f),0,Random.Range(11f,-11f));
            transform.eulerAngles = new Vector3(0,Random.Range(0f,360f),0);
            Enemy.Seeking = false;
            Enemy.transform.eulerAngles = new Vector3(0, Random.Range(0f, 360f), 0);
            Enemy.transform.localPosition = new Vector3(Random.Range(-14f, 0f), 0, Random.Range(11f, -11f));
            for (int i = 0; i<VectorActs.Count; i++) {
                VectorActs[i] = 0;
                Enemy.VectorActs[i] = 0;
            }
        }
    }

    public override void AgentAction(float[] vectorAction, string textAction) {
        if (Seeker && !Seeking)
            return;
        for (int i = 0; i<vectorAction.Length; i++) {
            VectorActs[i] = Mathf.Clamp(vectorAction[i],-1.0f,1.0f);
        }
        if (Seeking) {
            if (Seeker) {
                AddReward(-0.002f);
            } else {
                AddReward(0.002f);
            }
        } else {
            if (GetStepCount() >= 240) {
                Seeking = true;
                Enemy.Seeking = true;
            }
        }
        if (GetStepCount()>1500) {
            AddReward(1f);
            Enemy.AddReward(-1f);
            Enemy.Done();
            Done();
        }
    }

    private void FixedUpdate() {
        if (Seeker && !Seeking)
            return;
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

    private void OnCollisionEnter(Collision collision) {
        if (Seeking&&collision.transform.CompareTag("agent")&&!Seeker) {
            AddReward(-1f);
            Enemy.AddReward(1f);
            Enemy.Done();
            Done();
        }
    }
}
