using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BaseX;
using FrooxEngine;
using FrooxEngine.LogiX;
using Grpc.Core;
//using CodeX;
//using UnityEngine;
using FrooxEngine.LogiX;
using System.Threading;
using System.Linq;
using FrooxEngine.UIX;

namespace FrooxEngine.LogiX
{
    [OldNamespace("FrooxEngine")]
    [NodeName("To Array")]
    [Category(new string[] { "LogiX/VRAI" })]
    //public class ToList : MultiInputOperator<Input<float>, List<float>>
    public class ToArray : MultiInputOperator<float>
    {
        //public readonly List<float> Content
        public readonly Output<float[]> list;
        //public Sync<int> count;

        protected override void OnEvaluate()
        {
            base.OnEvaluate();
            //if (count.Value==default(int) || this.Operands.Count != count.Value)
            list.Value = new float[this.Operands.Count];
            for (int index = 0; index < this.Operands.Count; ++index)
            {
                list.Value[index] = this.Operands.GetElement(index).EvaluateRaw(default(float));
                //count.Value = this.Operands.Count;
            }
        }
        public override float Content
        {
            get
            {
                return default(float);
            }
        }
        protected override int MinInputs
        {
            get
            {
                return 0;
            }
        }
    }

    [OldNamespace("FrooxEngine")]
    [NodeName("Write To Array")]
    [Category(new string[] { "LogiX/VRAI" })]
    //public class ToList : MultiInputOperator<Input<float>, List<float>>
    public class WriteToArray : LogixNode
    {
        //public readonly List<float> Content
        public readonly Input<float> Value;
        public readonly Input<int> index;
        [AsOutput]
        public readonly Input<IValue<float[]>> Target;
        public readonly Impulse OnDone;
        public readonly Impulse OnFail;

        [ImpulseTarget]
        public void Write()
        {
            IValue<float[]> obj = this.Target.Evaluate((IValue<float[]>) null);
            if (obj != null)
            {
                if (obj is ISyncMember syncMember && syncMember.IsDriven && !syncMember.IsHooked)
                    return;
                obj.Value[this.index.Evaluate(default(int))] = this.Value.Evaluate(default(float));
                this.OnDone.Trigger();
            }
            else
                this.OnFail.Trigger();
        }
    }

    [OldNamespace("FrooxEngine")]
    [NodeName("Initalize Array")]
    [Category(new string[] { "LogiX/VRAI" })]
    //public class ToList : MultiInputOperator<Input<float>, List<float>>
    public class InitializeArray : LogixNode
    {
        //public readonly List<float> Content
        public readonly Input<float[]> Value;
        public readonly Input<int> count;
        [AsOutput]
        public readonly Input<IValue<float[]>> Target;
        public readonly Impulse OnDone;
        public readonly Impulse OnFail;

        [ImpulseTarget]
        public void Write()
        {
            IValue<float[]> obj = this.Target.Evaluate((IValue<float[]>) null);
            if (obj != null)
            {
                if (obj is ISyncMember syncMember && syncMember.IsDriven && !syncMember.IsHooked)
                    return;
                float[] init_array = this.Value.Evaluate(default(float[]));
                if (init_array != default(float[])) //if it is connected to something
                    obj.Value = init_array;
                else
                    obj.Value = new float[this.count.Evaluate(default(int))];
                this.OnDone.Trigger();
            }
            else
                this.OnFail.Trigger();
        }
    }

    [OldNamespace("FrooxEngine")]
    [NodeName("To Camera Array")]
    [Category(new string[] { "LogiX/VRAI" })]
    //public class ToList : MultiInputOperator<Input<float>, List<float>>
    public class ToCameraArray : MultiInputOperator<Camera>
    {
        //public readonly List<float> Content
        public readonly Output<Camera[]> list;

        protected override void OnEvaluate()
        {
            base.OnEvaluate();
            list.Value = new Camera[this.Operands.Count];
            for (int index = 0; index < this.Operands.Count; ++index)
                list.Value[index] = this.Operands.GetElement(index).EvaluateRaw(default(Camera));
        }
        public override Camera Content
        {
            get
            {
                return default(Camera);
            }
        }
        protected override int MinInputs
        {
            get
            {
                return 0;
            }
        }
    }


    [OldNamespace("FrooxEngine")]
    [NodeName("Get Element")]
    [Category(new string[] { "LogiX/VRAI" })]
    public class GetElement: LogixNode
    {
        public readonly Input<float[]> list;
        public readonly Input<int> index;
        public readonly Output<float> element;

        protected override void OnEvaluate()
        {
            base.OnEvaluate();
            float[] list = this.list.EvaluateRaw(new float[0]);
            int index = this.index.EvaluateRaw();
            if (list.Length == 0) element.Value = 0;
            else if (index >= list.Length)
            {
                index = list.Length - 1;
                element.Value = list[index];
            } else
            {
                element.Value = list[index];
            }
        }

    }
    [OldNamespace("FrooxEngine")]
    [NodeName("Unpack Array")]
    [Category(new string[] { "LogiX/VRAI" })]
    public class UnpackArray<T> : LogixNode
    {
        public readonly Input<T[]> input_array;
        public readonly SyncList<Output<T>> ValueOutputs;
        public readonly Output<int> OutputCount;

        protected override void OnAttach()
        {
            base.OnAttach();
            //this.ValueOutputs.Add();
            T[] raw1 = this.input_array.Evaluate(new T[0]);
            if (!(raw1 is null)) {
                this.OutputCount.Value = raw1.Length;
                int old_count = this.ValueOutputs.Count;
                this.ValueOutputs.EnsureExactCount(raw1.Length);
            }
            this.RefreshLogixBox();
        }

        //protected override bool AddElement()
        //{
        //    //this.ValueOutputs.Add();
        //    return true;
        //}

        //protected override bool RemoveElement()
        //{
        //    //if (this.ValueOutputs.Count == 0)
        //    //    return false;
        //    //this.ValueOutputs.RemoveAt(this.ValueOutputs.Count - 1);
        //    return true;
        //}

        protected override void OnChanges()
        {
            T[] raw1 = this.input_array.EvaluateRaw(new T[0]);
            if (!(raw1 is null))
            {
                this.OutputCount.Value = raw1.Length;
                int old_count = this.ValueOutputs.Count;
                this.ValueOutputs.EnsureExactCount(raw1.Length);
                if (old_count != this.OutputCount.Value)
                    this.RefreshLogixBox();
            }
        }

        protected override void OnEvaluate()
        {
            T[] raw1 = this.input_array.EvaluateRaw(new T[0]);
            if (!(raw1 is null)) {
                this.OutputCount.Value = raw1.Length;
                for (int index = 0; index < this.ValueOutputs.Count; ++index)
                    this.ValueOutputs[index].Value = raw1[index];
            }
            else
            {
                this.OutputCount.Value = this.ValueOutputs.Count;
            }
        }
        protected override System.Type FindOverload(NodeTypes connectingTypes)
        {
            System.Type type;
            if (!connectingTypes.inputs.TryGetValue("input_array", out type))
                return (System.Type)null;
            return typeof(UnpackArray<>).MakeGenericType(type.GetElementType());
        }

        //protected override void OnGenerateVisual(Slot root)
        //{
        //    UIBuilder ui = this.GenerateUI(root, 0.0f, 0.0f, float.MaxValue);
        //    ui.Panel();
        //    RectTransform footer;
        //    ui.HorizontalFooter(32f, out footer, out RectTransform _);
        //    UIBuilder uiBuilder = new UIBuilder(footer);
        //    uiBuilder.HorizontalLayout(4f, 0, new Alignment?());
        //    uiBuilder.Button("+", color.White, new ButtonEventHandler(this.AddOutput), 0.0f);
        //    uiBuilder.Button("-", color.White, new ButtonEventHandler(this.RemoveOutput), 0.0f);
        //}

        //private void AddOutput(IButton button, ButtonEventData eventData)
        //{
        //    this.ValueOutputs.Add();
        //    this.RefreshLogixBox();
        //}

        //private void RemoveOutput(IButton button, ButtonEventData eventData)
        //{
        //    if (this.ValueOutputs.Count <= 2)
        //        return;
        //    this.ValueOutputs.RemoveAt(this.ValueOutputs.Count - 1);
        //    this.RefreshLogixBox();
        //}


        //protected override bool OnInputConnect<I>(Input<I> input, IWorldElement output)
        //{
        //    T[] raw1 = this.input_array.EvaluateRaw(default(T[]));
        //    int old_count = this.ValueOutputs.Count;
        //    this.ValueOutputs.EnsureExactCount(raw1.Length);
        //    int new_count = this.ValueOutputs.Count;
        //    if (old_count != new_count) this.RefreshLogixBox();
        //    return true;
        //}

        //protected override void OnInputChange()
        //{
        //    T[] raw1 = this.input_array.EvaluateRaw(default(T[]));
        //    int old_count = this.ValueOutputs.Count;
        //    this.ValueOutputs.EnsureExactCount(raw1.Length);
        //    int new_count = this.ValueOutputs.Count;
        //    if (old_count != new_count) this.RefreshLogixBox();
        //}

        //internal override void OnSwapping(LogixNode oldNode)
        //{
        //    if (!(oldNode.GetSyncMember("ValueOutputs") is ISyncList syncMember))
        //        return;
        //    this.ValueOutputs.EnsureExactCount(syncMember.Count);
        //    this.RefreshLogixBox();
        //}


    }
    [OldNamespace("FrooxEngine")]
    [NodeName("array")]
    [NodeVisualType(typeof(float[]))]
    [Category(new string[] { "LogiX/VRAI" })]
    public class FloatArrayRegister : LogixOperator<float[]>, IValue<float[]>, IChangeable, IWorldElement
    {
        public readonly Sync<float[]> Value;

        public override float[] Content
        {
            get
            {
                return this.Value.Value;
            }
        }

        float[] IValue<float[]>.Value
        {
            get
            {
                return this.Value.Value;
            }
            set
            {
                this.Value.Value = value;
            }
        }
        public static FloatArrayRegister __New()
        {
            return new FloatArrayRegister();
        }
    }
        //[OldNamespace("FrooxEngine")]
        //[NodeName("floatTofloat")]
        //[Category(new string[] { "LogiX/Operators" })]
        //[Category(new string[] { "Hidden" })]
        //public class FloatToFloatList : Cast.CastNode<float, float[]>, Cast.ICastNode, IPassthroughNode, IComponent, IComponentBase, IDestroyable, IWorker, IWorldElement, IUpdatable, IChangeable, IAudioUpdatable, IInitializable, ILinkable
        //{
        //    public override float[] Content
        //    {
        //        get
        //        {
        //            float[] result = { this.In.EvaluateRaw() };
        //            return result;
        //        }
        //    }
        //}


        //[OldNamespace("FrooxEngine")]
        //[NodeName("To List")]
        ////[NodeOverload("AddMulti")]
        ////[HiddenNode]
        //[Category(new string[] { "LogiX/Operators" })]
        //public class ListToString : LogixOperator<float>
        //{
        //    public readonly Input<int> Index;
        //    public readonly Input<float> list;
        //    public override float Content
        //    {
        //        get
        //        {
        //            //return list.EvaluateRaw()[Index.EvaluateRaw()];
        //            return list.EvaluateRaw();
        //        }
        //    }
        //}
    }
