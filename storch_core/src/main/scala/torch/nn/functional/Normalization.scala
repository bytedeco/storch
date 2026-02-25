package torch
package nn
package functional

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  TensorOptional,
  LocalResponseNormOptions,
  NormalizeFuncOptions,
  DoubleOptional
}
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.fromNative

private[torch] trait Normalization {
  def local_response_norm[D <: DType](
      input: Tensor[D],
      size: Long,
      alpha: Double = 0.0001,
      beta: Double = 0.75,
      k: Double = 0.1
  ): Tensor[D] = {
    val options = new LocalResponseNormOptions(size)
    options.alpha().put(alpha)
    options.beta().put(beta)
    options.k().put(k)
    val result = torchNative.local_response_norm(input.native, options)
    fromNative(result)
  }

  def normalize[D <: DType](
      input: Tensor[D],
      p: Double = 2.0,
      dim: Long = 1,
      eps: Double = 1e-12,
      out: Tensor[D]
  ): Tensor[D] = {
    val options: NormalizeFuncOptions = new NormalizeFuncOptions()
    options.dim().put(dim)
    options.eps().put(eps)
    options.p().put(p)
    options.out().put(TensorOptional(out.native))
    val result = torchNative.normalize(input.native, options)
    fromNative(result)
  }

  def rms_norm[D1 <: DType](
      input: Tensor[D1],
      normalized_shape: Seq[Long],
      weight: Option[Tensor[Float32]],
      eps: Option[Double]
  ): Tensor[D1] = {
    val weightOpt =
      if weight.isDefined then new TensorOptional(weight.get.native) else new TensorOptional()
    val epsOpt = if eps.isDefined then new DoubleOptional(eps.get) else new DoubleOptional()
    fromNative(torchNative.rms_norm(input.native, normalized_shape.toArray, weightOpt, epsOpt))
  }
}
