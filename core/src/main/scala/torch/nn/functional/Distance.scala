package torch
package nn
package functional

//import Derive.derive
//import org.bytedeco.pytorch
//GroupNormFuncOptions,
//InstanceNormFuncOptions,
//LayerNormFuncOptions,
//LocalResponseNormOptions,
//NormalizeFuncOptions,
//ScalarTypeOptional
import org.bytedeco.pytorch.{PairwiseDistanceOptions, CosineSimilarityOptions}
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.fromNative
import org.bytedeco.javacpp.annotation.{ByRef, ByVal, Const, Namespace}

private[torch] trait Distance {

  def pairwise_distance[D <: DType](x1: Tensor[D], x2: Tensor[D], p: Double = 2.0): Tensor[D] = {
    val options: PairwiseDistanceOptions = new PairwiseDistanceOptions()
    options.p().put(2.0)
    val result = torchNative.pairwise_distance(x1.native, x2.native, options)
    fromNative(result)
  }

  def cosine_similarity[D <: DType](
      x1: Tensor[D],
      x2: Tensor[D],
      dim: Int = 1,
      eps: Double = 1e-8
  ): Tensor[D] = {

    val options = new CosineSimilarityOptions()
    options.dim().put(dim)
    options.eps().put(eps)
    val result = torchNative.cosine_similarity(x1.native, x2.native, options)
    fromNative(result)
  }

  def pdist[D <: DType](x1: Tensor[D], p: Double = 2.0): Tensor[D] = {
    val result = torchNative.pdist(x1.native, p)
    fromNative(result)
  }

}
