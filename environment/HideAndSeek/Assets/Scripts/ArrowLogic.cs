using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ArrowLogic : MonoBehaviour{
    public ArcherAI Sender;
    float TimeOut = 4;
    public System.Action hitAction;

    private void Update() {
        if (TimeOut>0) {
            TimeOut -= Time.deltaTime;
        } else {
            Destroy(gameObject);
        }
        if (transform.localPosition.sqrMagnitude > 5000) {
            Destroy(gameObject);
        }
    }

    private void OnCollisionEnter(Collision collision) {
        if (transform.localPosition.sqrMagnitude > 5000 || collision.collider.tag=="wall") {
            Destroy(gameObject);
        }
        if (collision.collider.tag=="Player") {
            collision.collider.transform.parent.parent.GetComponent<ArcherAI>().HitOrShot(true);
            Sender.HitOrShot(false);
            transform.parent = collision.collider.transform;
        }
        hitAction();
        Destroy(GetComponent<Rigidbody>());
        Destroy(GetComponent<BoxCollider>());
        Destroy(this);
    }
}
