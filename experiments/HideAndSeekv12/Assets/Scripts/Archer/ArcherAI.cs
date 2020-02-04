using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class ArcherAI : Agent{
    Rigidbody rb;
    Rigidbody erb;
    float Horizontal = 0;
    float Vertical = 0;
    float[] acts;
    Transform Scale;
    Vector3 CurrentRotation;
    public GameObject Arrow;
    Vector3 StartPos;
    Quaternion StartRot;
    Quaternion StartRotPiv;
    List<GameObject> SentArrows = new List<GameObject>();
    public ArcherAI Enemy;
    Transform TEnemy;

    [HideInInspector]
    public Transform Pivot;
    [HideInInspector]
    public float Height = 1;
    [HideInInspector]
    public float TimeOut = 0;
    public GameObject LastArrow;

    // Start is called before the first frame update
    void Start(){
        rb = GetComponent<Rigidbody>();
        Pivot = transform.GetChild(0);
        Scale = transform.GetChild(1);
        CurrentRotation = new Vector3(Pivot.localEulerAngles.x-360, transform.localEulerAngles.y, 0);
        StartPos = transform.position;
        StartRot = transform.rotation;
        StartRotPiv = Pivot.rotation;
        TEnemy = Enemy.transform;
        erb = TEnemy.GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void Update(){
        Height = Mathf.Lerp(Height, ((Mathf.Clamp(acts[4], -1.0f, 1.0f) + 1) / 4) + 0.5f, 6f * Time.deltaTime);
        Horizontal = Mathf.Lerp(Horizontal, Mathf.Clamp(acts[1], -1.0f, 1.0f)*0.5f, 8f*Time.deltaTime);
        Vertical = Mathf.Lerp(Vertical, Mathf.Clamp(acts[0], -1.0f, 1.0f), 8f * Time.deltaTime);
        //Slows movement when ducking
        rb.velocity = (transform.forward * Vertical + transform.right * Horizontal)*10*(1.5f*Height-0.5f) + Vector3.up*rb.velocity.y;
        CurrentRotation += new Vector3(Mathf.Clamp(acts[2], -1.0f, 1.0f)*5, Mathf.Clamp(acts[3], -1.0f, 1.0f)*6, 0);
        if (CurrentRotation.x > 90) {
            CurrentRotation.x = 90;
        } else if (CurrentRotation.x < -90) {
            CurrentRotation.x = -90;
        }
        Pivot.localEulerAngles = new Vector3(CurrentRotation.x,0,0);
        transform.localEulerAngles = new Vector3(0, CurrentRotation.y, 0);
        Scale.transform.localScale = new Vector3(1, Height, 1);
        Pivot.transform.localPosition = new Vector3(0,1.65f*Height,0);
        if (TimeOut>0) {
            TimeOut -= Time.deltaTime;
        }
    }

    public override void AgentAction(float[] vectorAction) {
        acts = vectorAction;
        if (vectorAction[5] > 0)
            Shoot();
    }

    public override void CollectObservations() {
        AddVectorObs(TEnemy.localPosition.x);
        AddVectorObs(TEnemy.localPosition.z);
        AddVectorObs(erb.velocity.x);
        AddVectorObs(erb.velocity.z);
        AddVectorObs(Enemy.Height);
        AddVectorObs(Enemy.Pivot.forward);
        AddVectorObs(Enemy.TimeOut/2f);
        if (Enemy.LastArrow!=null) {
            AddVectorObs(Enemy.LastArrow.transform.localPosition);
            AddVectorObs(Enemy.LastArrow.transform.forward);
        } else {
            AddVectorObs(Vector3.one*200);
            AddVectorObs(Vector3.zero);
        }
        AddVectorObs(transform.localPosition.x);
        AddVectorObs(transform.localPosition.z);
        AddVectorObs(rb.velocity.x);
        AddVectorObs(rb.velocity.z);
        AddVectorObs(Height);
        AddVectorObs(Pivot.forward);
        AddVectorObs(TimeOut/2f);
        if (LastArrow != null) {
            AddVectorObs(LastArrow.transform.localPosition);
            AddVectorObs(LastArrow.transform.forward);
        } else {
            AddVectorObs(Vector3.one * 200);
            AddVectorObs(Vector3.zero);
        }
    }

    public override float[] Heuristic() {
        return base.Heuristic();
        //return new float[] { (Input.GetKey(KeyCode.W)?1:(Input.GetKey(KeyCode.S)?-1:0)), Input.GetKey(KeyCode.A) ? -1 : (Input.GetKey(KeyCode.D) ? 1 : 0), Input.GetKey(KeyCode.Q) ? -1 : (Input.GetKey(KeyCode.E) ? 1 : 0), Input.GetKey(KeyCode.R) ? -1 : (Input.GetKey(KeyCode.F) ? 1 : 0)
        //,Input.GetKey(KeyCode.G) ? -1 : (Input.GetKey(KeyCode.H) ? 1 : 0),Input.GetKey(KeyCode.Space) ? 1:0};
    }

    public void HitOrShot(bool hit) {
        if (hit) {
            //we got hit
            AddReward(-1.0f);
        } else {
            //we hit the other
            AddReward(1.0f);
        }
    }

    void Shoot() {
        if (TimeOut > 0)
            return;
        GameObject a = Instantiate(Arrow,Arrow.transform.position,Arrow.transform.rotation);
        a.transform.parent = transform.parent;
        a.SetActive(true);
        a.GetComponent<Rigidbody>().velocity = a.transform.forward * 100;
        a.GetComponent<ArrowLogic>().Sender = this;
        a.GetComponent<ArrowLogic>().hitAction = delegate () { LastArrow = null; };
        TimeOut = 2;
        SentArrows.Add(a);
        LastArrow = a;
    }

    public override void AgentReset() {
        rb.velocity = Vector3.zero;
        transform.position = StartPos;
        transform.rotation = StartRot;
        Pivot.rotation = StartRotPiv;
        CurrentRotation = new Vector3(Pivot.localEulerAngles.x - 360, transform.localEulerAngles.y, 0);
        Horizontal = 0;
        Vertical = 0;
        Height = 1;
        acts = new float[]{0,0,0,0,0,0};
        for (int i = 0; i<SentArrows.Count; i++) {
            if (SentArrows[i] != null)
                Destroy(SentArrows[i]);
        }
        SentArrows.Clear();
        LastArrow = null;
        Enemy.Done();
    }
}
