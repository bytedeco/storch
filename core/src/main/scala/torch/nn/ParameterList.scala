package torch
package nn

import org.bytedeco.pytorch.{ParameterListImpl}
import torch.internal.NativeConverters.fromNative

class ParameterList[D <: DType] extends ParameterListImpl {

  override def reset(): Unit = super.reset()

  override def pretty_print(stream: _root_.org.bytedeco.javacpp.Pointer): Unit =
    super.pretty_print(stream)

  def append(param: Tensor[D]): Unit = super.append(param.native)

  override def append(pair: _root_.org.bytedeco.pytorch.StringTensorDictItem): Unit =
    super.append(pair)

  override def begin(): _root_.org.bytedeco.pytorch.StringTensorDictItemVector.Iterator =
    super.begin()

  override def end(): _root_.org.bytedeco.pytorch.StringTensorDictItemVector.Iterator = super.end()

  def at_native(idx: Long): Tensor[D] = fromNative(super.at(idx))

  def get_native(idx: Long): Tensor[D] = fromNative(super.get(idx))

  override def size(): Long = super.size()

  override def is_empty(): Boolean = super.is_empty()
}
