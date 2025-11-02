package torch
package nn
package functional

import Derive.derive
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  PadFuncOptions,
  LongVector,
  kConstant,
  kReflect,
  kReplicate,
  kCircular,
  DoubleVector,
  LongVectorOptional,
  DoubleVectorOptional,
  InterpolateMode,
  PixelShuffleOptions,
  InterpolateFuncOptions,
  InstanceNormFuncOptions,
  LayerNormFuncOptions,
  LocalResponseNormOptions,
  NormalizeFuncOptions,
  ScalarTypeOptional
}
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.fromNative
import org.bytedeco.javacpp.annotation.{ByRef, ByVal, Const, Namespace}
private[torch] trait Padding {
  enum PaddingMode(val code: Int):
    case Reflect extends PaddingMode(0)
    case Replicate extends PaddingMode(1)
    case Circular extends PaddingMode(2)
    case Constant extends PaddingMode(3)

  // torch.nn.functional.pad(input, pad, mode='constant', value=None)
  def pad[D <: DType](
      input: Tensor[D],
      pad: Seq[Long],
      mode: PaddingMode = PaddingMode.Constant,
      value: Option[Double] = None
  ): Tensor[D] = {

    val padVector = new LongVector(pad*)
    val options = new PadFuncOptions(padVector)
    val nativeMode = mode match {
      case PaddingMode.Reflect   => new kReflect()
      case PaddingMode.Replicate => new kReplicate()
      case PaddingMode.Circular  => new kCircular()
      case PaddingMode.Constant  => new kConstant()
    }
    options.mode.put(nativeMode)
    if value.isDefined then options.value.put(value.get) else options.value.put(0d)
    val result = torchNative.pad(input.native, options)
    fromNative(result)
  }

//  def pad[D <: DType](x: Tensor[D], output_size: List[Long]): Tensor[D] = {
//
//    val outputRef = LongArrayRef(output_size.toArray, output_size.length)
//    val result = torchNative.pad(x.native, outputRef)
//    fromNative(result)
//  }
}
