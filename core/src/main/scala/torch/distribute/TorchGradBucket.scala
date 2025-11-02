package torch
package distribute

import org.bytedeco.pytorch.global.torch.BuiltinCommHookType
import org.bytedeco.pytorch.{
  Reducer,
  TensorVector,
  LongArrayRefVector,
  LongArrayRef,
  LongVector,
  SizeTVector,
  SizeTVectorVector,
  BoolVector,
  CommHookInterface,
  TensorOptional,
  GradBucket,
  StringTensorMap,
  FutureList,
  Logger,
  ProcessGroup
}
import torch.internal.NativeConverters.fromNative

import scala.collection.mutable.ListBuffer

abstract class TorchGradBucket {

  val native: GradBucket

  def getSparseGradIndices = native.getSparseGradIndices()

  def isLast: Boolean = native.isLast()

  def getParameters: Seq[Tensor[?]] = {
    val vector = native.getParameters()
    val buffer = new ListBuffer[Tensor[?]]()
    var it = vector.begin()
    while (!it.equals(vector.end())) {
      buffer.append(from_native(it.get()))
      it = it.increment()
    }
    buffer.toSeq
  }

  def getGradients: Seq[Tensor[?]] = {
    val vector = native.getGradients()
    val buffer = new ListBuffer[Tensor[?]]()
    var it = vector.begin()
    while (!it.equals(vector.end())) {
      buffer.append(from_native(it.get()))
      it = it.increment()
    }
    buffer.toSeq
  }

  def getBufferRef: Tensor[?] = fromNative(native.getBufferRef())

  def getBuffer: Tensor[?] = fromNative(native.getBuffer())

  def getIndex: Long = native.getIndex()

  def setBuffer(tensor: Tensor[?]) = native.setBuffer(tensor.native)

  def getGradientBucket(
      index: Long,
      bucket_count: Long,
      tensor: Tensor[?],
      offsets: Array[Long],
      lengths: Array[Long],
      sizes_vec: Array[Long],
      parameters: Seq[Tensor[?]],
      sparse_grad_indice: Option[Tensor[?]]
  ): GradBucket = {
    val offsetNative = new SizeTVector(offsets*)
    val lengthsNative = new SizeTVector(lengths*)
    val size: Array[LongArrayRef] = sizes_vec.map(el => new LongArrayRef(el))
    val sizeVec = new LongArrayRefVector(size*)
    val parametersNative = new TensorVector(parameters.map(_.native)*)
    val sparseGradIndice =
      if sparse_grad_indice.isDefined then new TensorOptional(sparse_grad_indice.get.native)
      else new TensorOptional()
    new GradBucket(
      index,
      bucket_count,
      tensor.native,
      offsetNative,
      lengthsNative,
      sizeVec,
      parametersNative,
      sparseGradIndice
    )

  }
}
