package torch
package nn
import org.bytedeco.pytorch.ParameterDictImpl
//import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.fromNative

import scala.collection.mutable.ListBuffer

class ParameterDict[D <: DType] extends ParameterDictImpl {

  override def reset(): Unit = super.reset()

  override def pretty_print(stream: _root_.org.bytedeco.javacpp.Pointer): Unit =
    super.pretty_print(stream)

  def insert(key: String, param: Tensor[D]): Tensor[D] =
    fromNative[D](super.insert(key, param.native))

  def pop_native(key: String): Tensor[D] = fromNative[D](super.pop(key))

  def keys_native(): Seq[String] = {
    val keyVector = super.keys()
    val buffer = new ListBuffer[String]()
    var it = keyVector.begin()
    while !it.equals(keyVector.end()) do
      val element: String = it.get().toString()
      buffer.append(element)
      it = it.increment()
    buffer.toSeq
  }

  def values_native(): Seq[Tensor[D]] = {
    val tensorVector = super.values()
    val buffer = new ListBuffer[Tensor[D]]()
    var it = tensorVector.begin()
    while !it.equals(tensorVector.end()) do
      val element = it.get()
      buffer.append(fromNative[D](element))
      it = it.increment()
    buffer.toSeq
  }

  override def begin(): _root_.org.bytedeco.pytorch.StringTensorDictItemVector.Iterator =
    super.begin()

  override def end(): _root_.org.bytedeco.pytorch.StringTensorDictItemVector.Iterator = super.end()

  override def size(): Long = super.size()

  override def empty(): Boolean = super.empty()

  override def clear(): Unit = super.clear()

  override def contains(key: String): Boolean = super.contains(key)

  def get_native(key: String): Tensor[D] = fromNative(super.get(key))
}

//  override def insert(key: _root_.org.bytedeco.javacpp.BytePointer, param: _root_.org.bytedeco.pytorch.Tensor): _root_.org.bytedeco.pytorch.Tensor = super.insert(key, param)

//  override def pop(key: _root_.org.bytedeco.javacpp.BytePointer): _root_.org.bytedeco.pytorch.Tensor = super.pop(key)
