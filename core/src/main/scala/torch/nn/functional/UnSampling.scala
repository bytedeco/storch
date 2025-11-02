package torch
package nn
package functional

//import org.bytedeco.pytorch.{
//  StringViewOptional,
//  LongArrayRefOptional,
//  LongOptional,
//  BoolOptional,
//  SymInt,
//  SymIntOptional
//}
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.fromNative
//import org.bytedeco.javacpp.annotation.{ByRef, ByVal, Const, Namespace}

trait UnSampling {

  def upsample_trilinear3d[D <: DType](
      x: Tensor[D],
      output_size: List[Long],
      align_corners: Boolean,
      scale_factors: List[Double]
  ): Tensor[D] = {
    val result =
      torchNative.upsample_trilinear3d(x.native, output_size.toArray, align_corners, scale_factors*)
    fromNative(result)
  }

  def upsample_linear1d[D <: DType](
      x: Tensor[D],
      output_size: List[Long],
      align_corners: Boolean,
      scale_factors: List[Double]
  ): Tensor[D] = {
    val result =
      torchNative.upsample_linear1d(x.native, output_size.toArray, align_corners, scale_factors*)
    fromNative(result)

  }

  def upsample_bicubic2d[D <: DType](
      x: Tensor[D],
      output_size: List[Long],
      align_corners: Boolean,
      scale_factors: List[Double]
  ): Tensor[D] = {
    val result =
      torchNative.upsample_bicubic2d(x.native, output_size.toArray, align_corners, scale_factors*)
    fromNative(result)
  }

  def upsample_nearest1d[D <: DType](
      x: Tensor[D],
      output_size: List[Long],
      scale_factors: List[Double]
  ): Tensor[D] = {
    val result = torchNative.upsample_nearest1d(x.native, output_size.toArray, scale_factors*)
    fromNative(result)

  }

  def upsample_nearest2d[D <: DType](
      x: Tensor[D],
      output_size: List[Long],
      scale_factors: List[Double]
  ): Tensor[D] = {
    val result = torchNative.upsample_nearest2d(x.native, output_size.toArray, scale_factors*)
    fromNative(result)
  }

  def upsample_nearest3d[D <: DType](
      x: Tensor[D],
      output_size: List[Long],
      scale_factors: List[Double]
  ): Tensor[D] = {
    val result = torchNative.upsample_nearest3d(x.native, output_size.toArray, scale_factors*)
    fromNative(result)
  }

  //
  def upsample_bilinear2d[D <: DType](
      x: Tensor[D],
      output_size: List[Long],
      align_corners: Boolean,
      scale_factors: List[Double]
  ): Tensor[D] = {
    val result =
      torchNative.upsample_bilinear2d(x.native, output_size.toArray, align_corners, scale_factors*)
    fromNative(result)
  }
}
