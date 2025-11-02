/*
 * Copyright 2022 storch.dev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package torch
package nn
package functional

import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.fromNative
import org.bytedeco.pytorch.{
  kSum,
  kMean,
  kMax,
  DoubleOptional,
  LongOptional,
  EmbeddingBagFuncOptions,
  EmbeddingFuncOptions
}
//  EmbeddingBagMode,
//import org.bytedeco.pytorch.{
//  BatchNormFuncOptions,
//  GroupNormFuncOptions,
//  InstanceNormFuncOptions,
//  LayerNormFuncOptions,
//  LocalResponseNormOptions,
//  NormalizeFuncOptions,
//  ScalarTypeOptional
//}
//import org.bytedeco.javacpp.annotation.{ByRef, ByVal, Const, Namespace}

private[torch] trait Sparse {

  /** Takes LongTensor with index values of shape `(*)` and returns a tensor of shape `(*,
    * numClasses)` that have zeros everywhere except where the index of last dimension matches the
    * corresponding value of the input tensor, in which case it will be 1.
    *
    * @group nn_sparse
    */
  def oneHot(input: Tensor[Int64], numClasses: Long = -1): Tensor[Int64] =
    fromNative(torchNative.one_hot(input.native, numClasses))

  def one_hot(input: Tensor[Int64], numClasses: Long = -1): Tensor[Int64] =
    oneHot(input, numClasses)

  def embedding_bag[D <: DType](
      input: Tensor[D],
      weight: Tensor[D],
      offsets: Tensor[D],
      max_norm: Double,
      norm_type: Double = 2,
      scale_grad_by_freq: Boolean = false,
      mode: String = "mean",
      sparse: Boolean = false,
      per_sample_weight: Tensor[D],
      include_last_offset: Boolean = false,
      padding_idx: Long
  ): Tensor[D] = {
    val options = EmbeddingBagFuncOptions()
    val bagModeNative = mode match {
      case "sum" | "Sum"   => new kSum
      case "mean" | "Mean" => new kMean
      case "max" | "Max"   => new kMax
//      case "none"|"None" => EmbeddingBagMode.None
    }
    options.offsets.put(offsets.native)
    options.max_norm().put(DoubleOptional(max_norm))
    options.norm_type().put(norm_type)
    options.scale_grad_by_freq().put(scale_grad_by_freq)
    options.mode().put(bagModeNative)
    options.sparse().put(sparse)
    options.per_sample_weights.put(per_sample_weight.native)
    options.include_last_offset().put(include_last_offset)
    options.padding_idx().put(LongOptional(padding_idx))
    fromNative(torchNative.embedding_bag(input.native, weight.native, options))
  }

  def embedding[D <: DType](
      input: Tensor[D],
      weight: Tensor[D],
      padding_idx: Long,
      max_norm: Double,
      norm_type: Double = 2.0,
      scale_grad_by_freq: Boolean = false,
      sparse: Boolean = false
  ): Tensor[D] = {
    val options = EmbeddingFuncOptions()
    options.padding_idx().put(padding_idx)
    options.max_norm().put(max_norm)
    options.norm_type().put(norm_type)
    options.scale_grad_by_freq().put(scale_grad_by_freq)
    options.sparse().put(sparse)
    fromNative(torchNative.embedding(input.native, weight.native, options))
  }

}
