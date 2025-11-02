package torch
package ops

import org.bytedeco.javacpp.Pointer
import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.pytorch.{
  AutoFwGradMode,
  AutoGradMode,
  AutogradContext,
  AutogradMetaFactory,
  AutogradState,
  ForwardGrad,
  GradBucket,
  GradCallback,
  GradMode,
  LongArrayRefVector,
  NoGradGuard,
  PostAccumulateGradHook,
  SizeTVector,
  TensorOptional,
  BoolOptional,
  TensorVector
}

trait GradOps {

  /** Creates a new GradCallback. // PostAccumulateGradHook
    * @return
    */
  def gradCallback(): GradCallback = {

    val gradCallback = new GradCallback(new Pointer())
    gradCallback
  }
  def autoFwGradMode(enabled: Boolean = true): AutoFwGradMode = {
    val autoFwGradMode = new AutoFwGradMode(enabled)
    autoFwGradMode
  }

  def gradBucket[P <: DType](
      index: Long,
      bucketCount: Long,
      tensor: Tensor[P],
      offsets: SizeTVector,
      lengths: SizeTVector,
      sizesVec: LongArrayRefVector,
      parameters: TensorVector,
      sparseGradIndices: TensorOptional
  ): GradBucket = {
    val gradBucket = new GradBucket(
      index,
      bucketCount,
      tensor.native,
      offsets,
      lengths,
      sizesVec,
      parameters,
      sparseGradIndices
    )
    gradBucket
  }

  /** Creates a new AutogradState with the given flags.
    *
    * @param gradMode
    * @param inferenceMode
    * @param fwGradMode
    * @param multiThreadingEnabled
    * @return
    *   // AutogradState.get_tls_state() // AutoFwGradMode
    */
  def autogradState(
      gradMode: Boolean,
      inferenceMode: Boolean,
      fwGradMode: Boolean,
      multiThreadingEnabled: Boolean
  ): AutogradState = {
    val auto = new AutogradState(gradMode, inferenceMode, fwGradMode, multiThreadingEnabled)
    auto
  }

  /** Creates a new ForwardGrad.
    *
    * @return
    *   // ForwardGrad.undef_grad()
    */
  def forwardGrad(): ForwardGrad = {
    val forwardGrad = new ForwardGrad()
    forwardGrad
  }

  def undef_grad() = {
    ForwardGrad.undef_grad()
  }

  def autogradContext(): AutogradContext = {
    val autogradContext = new AutogradContext()
    autogradContext
  }

  def require_grad(flag: Boolean) = torchNative.requires_grad(flag)

  def no_grad_raw(): NoGradGuard = {
    val noGradGuard = new NoGradGuard()
    noGradGuard
  }

  def enable_grad(enable: Boolean = true): Unit = {
    GradMode.set_enabled(enable)
  }

  def disable_grad(disable: Boolean = true): Unit = {
    val flag = !disable
    GradMode.set_enabled(flag)
  }

  def is_enable(): Boolean = {
    GradMode.is_enabled()
  }

  object autograd {

    /** Creates a new GradCallback. // PostAccumulateGradHook
      *
      * @return
      */
    def gradCallback(): GradCallback = {

      val gradCallback = new GradCallback(new Pointer())
      gradCallback
    }

    def autoFwGradMode(enabled: Boolean = true): AutoFwGradMode = {
      val autoFwGradMode = new AutoFwGradMode(enabled)
      autoFwGradMode
    }

    def gradBucket[P <: DType](
        index: Long,
        bucketCount: Long,
        tensor: Tensor[P],
        offsets: SizeTVector,
        lengths: SizeTVector,
        sizesVec: LongArrayRefVector,
        parameters: TensorVector,
        sparseGradIndices: TensorOptional
    ): GradBucket = {
      val gradBucket = new GradBucket(
        index,
        bucketCount,
        tensor.native,
        offsets,
        lengths,
        sizesVec,
        parameters,
        sparseGradIndices
      )
      gradBucket
    }

    /** Creates a new AutogradState with the given flags.
      *
      * @param gradMode
      * @param inferenceMode
      * @param fwGradMode
      * @param multiThreadingEnabled
      * @return
      *   // AutogradState.get_tls_state() // AutoFwGradMode
      */
    def autogradState(
        gradMode: Boolean,
        inferenceMode: Boolean,
        fwGradMode: Boolean,
        multiThreadingEnabled: Boolean
    ): AutogradState = {
      val auto = new AutogradState(gradMode, inferenceMode, fwGradMode, multiThreadingEnabled)
      auto
    }

    /** Creates a new ForwardGrad.
      *
      * @return
      *   // ForwardGrad.undef_grad()
      */
    def forwardGrad(): ForwardGrad = {
      val forwardGrad = new ForwardGrad()
      forwardGrad
    }

    def undef_grad() = {
      ForwardGrad.undef_grad()
    }

    def autogradContext(): AutogradContext = {
      val autogradContext = new AutogradContext()
      autogradContext
    }

    def require_grad(flag: Boolean) = torchNative.requires_grad(flag)

    def no_grad_raw(): NoGradGuard = {
      val noGradGuard = new NoGradGuard()
      noGradGuard
    }

    def enable_grad(enable: Boolean = true): Unit = {
      GradMode.set_enabled(enable)
    }

    def disable_grad(disable: Boolean = true): Unit = {
      val flag = !disable
      GradMode.set_enabled(flag)
    }

    def is_enable(): Boolean = {
      GradMode.is_enabled()
    }

    def backward[D <: DType](tensors: Seq[Tensor[D]] | Tensor[D]): Unit = {
      val tensorsSeq = tensors match {
        case t: Tensor[D]       => Seq(t)
        case ts: Seq[Tensor[D]] => ts
      }
      torchNative.backward(
        new TensorVector(tensorsSeq.map(_.native)*)
      )
    }

    def backward[D1 <: DType, D2 <: DType](
        tensors: Seq[Tensor[D1]] | Tensor[D1],
        grad_tensors: Seq[Tensor[D2]] | Tensor[D2] | Option[Tensor[D2]] = None,
        retain_graph: Option[Boolean] | Boolean = None,
        create_graph: Boolean = false,
        inputs: Seq[Tensor[Promoted[D1, D2]]]
    ): Unit = {
      val tensorsSeq = tensors match {
        case t: Tensor[D1]       => Seq(t)
        case ts: Seq[Tensor[D1]] => ts
      }
      val grad_tensorsSeq = grad_tensors match {
        case Some(t: Tensor[D2])       => Seq(t)
        case Some(ts: Seq[Tensor[D2]]) => ts
        case None                      => Seq.empty
      }
      val tensorVector = torchNative.backward(
        new TensorVector(tensorsSeq.map(_.native)*),
        new TensorVector(grad_tensorsSeq.map(_.native)*),
        retain_graph match {
          case Some(b: Boolean) => new BoolOptional(b)
          case None             => new BoolOptional()
          case b: Boolean       => new BoolOptional(b)
        },
        create_graph,
        new TensorVector(inputs.map(_.native)*)
      )
    }

    def grad[D1 <: DType, D2 <: DType](
        outputs: Seq[Tensor[D1]] | Tensor[D1],
        inputs: Seq[Tensor[D2]] | Tensor[D2]
    ): Seq[Tensor[Promoted[D1, D2]]] = {
      val outputsSeq = outputs match {
        case t: Tensor[D1]       => Seq(t)
        case ts: Seq[Tensor[D1]] => ts
      }
      val inputsSeq = inputs match {
        case t: Tensor[D2]       => Seq(t)
        case ts: Seq[Tensor[D2]] => ts
      }
      val vec = torchNative.grad(
        new TensorVector(outputsSeq.map(_.native)*),
        new TensorVector(inputsSeq.map(_.native)*)
      )
      tensorVectorToSeqTensor2(vec)
    }

    def grad[D1 <: DType, D2 <: DType, D3 <: DType](
        outputs: Seq[Tensor[D1]] | Tensor[D1],
        inputs: Seq[Tensor[D2]] | Tensor[D2],
        grad_outputs: Seq[Tensor[D3]] | Tensor[D3] | Option[Tensor[D3]] = None,
        retain_graph: Option[Boolean] | Boolean = None,
        create_graph: Boolean = false,
        allow_unused: Boolean = false
    ): Seq[Tensor[Promoted[D3, D2]]] = {
      val outputsSeq = outputs match {
        case t: Tensor[D1]       => Seq(t)
        case ts: Seq[Tensor[D1]] => ts
      }
      val inputsSeq = inputs match {
        case t: Tensor[D2]       => Seq(t)
        case ts: Seq[Tensor[D2]] => ts
      }
      val grad_outputsSeq = grad_outputs match {
        case Some(t: Tensor[D3]) => Seq(t)
        case ts: Seq[Tensor[D3]] => ts
        case t: Tensor[D3]       => Seq(t)
        case None                => Seq.empty[Tensor[D3]]
      }
      val nativeRetainGraph = retain_graph match {
        case Some(b: Boolean) => new BoolOptional(b)
        case None             => new BoolOptional()
        case b: Boolean       => new BoolOptional(b)
      }
      val tensorVector = torchNative.grad(
        new TensorVector(outputsSeq.map(_.native)*),
        new TensorVector(inputsSeq.map(_.native)*),
        new TensorVector(grad_outputsSeq.map(_.native)*),
        nativeRetainGraph,
        create_graph,
        allow_unused
      )
      tensorVectorToSeqTensor2(tensorVector)
    }

    def enter_dual_level(): Long = {
      torchNative.enter_dual_level()
    }

    def exit_dual_level(level: Long): Unit = {
      torchNative.exit_dual_level(level)
    }

  }

}
