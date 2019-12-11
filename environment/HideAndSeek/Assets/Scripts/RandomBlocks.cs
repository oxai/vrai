using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RandomBlocks : MonoBehaviour{
    public Transform BlockParent;
    public List<GameObject> Blocks;
    public int Number = 10;

    // Update is called once per frame
    void Update(){
        if (Input.GetKeyDown(KeyCode.G)) {
            for (int i = 0; i<Blocks.Count; i++) {
                Destroy(Blocks[i]);
            }
            Blocks.Clear();
            for (int i = 0; i<Number; i++) {
                GameObject T = GameObject.CreatePrimitive(PrimitiveType.Cube);
                T.transform.parent = BlockParent;
                T.transform.localPosition = new Vector3(Random.Range(-50f,50f),0, Random.Range(-50f, 50f));
                T.transform.eulerAngles = new Vector3(0,Random.Range(0f,360f),0);
                T.transform.localScale = new Vector3(Random.Range(1f,8f),2,Random.Range(1f,8f));
                Blocks.Add(T);
            }
        }
    }
}
