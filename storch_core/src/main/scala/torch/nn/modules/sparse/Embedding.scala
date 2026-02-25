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
package modules
package sparse

import org.bytedeco.pytorch
import org.bytedeco.pytorch.EmbeddingImpl
import org.bytedeco.pytorch.EmbeddingOptions
import torch.internal.NativeConverters.{fromNative, toNative}

// format: off
/** A simple lookup table that stores embeddings of a fixed dictionary and size.
  *
  * This module is often used to store word embeddings and retrieve them using indices. The input to
  * the module is a list of indices, and the output is the corresponding word embeddings.
  *
  * @group nn_sparse
  *
  * @param numEmbeddings
  *   Size of the dictionary of embeddings
  * @param embeddingDim
  *   The size of each embedding vector
  * @param paddingIdx
  *   If specified, the entries at `paddingIdx` do not contribute to the gradient; therefore, the
  *   embedding vector at `paddingIdx` is not updated during training, i.e. it remains as a fixed
  *   "pad". For a newly constructed Embedding, the embedding vector at `paddingIdx` will default to
  *   all zeros, but can be updated to another value to be used as the padding vector.
  * @param maxNorm
  *   If given, each embedding vector with norm larger than `maxNorm` is renormalized to have norm
  *   `maxNorm`.
  * @param normType
  *   The p of the p-norm to compute for the `maxNorm` option. Default `2`.
  * @param scaleGradByFreq
  *   If given, this will scale gradients by the inverse of frequency of the words in the
  *   mini-batch. Default `false`.
  * @param sparse
  *   If ``True``, gradient w.r.t. `weight` matrix will be a sparse tensor. See Notes for more
  *   details regarding sparse gradients.
  * 
  * @see See [[https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_embedding.html#class-embedding Pytorch C++ Embedding]]
  * @see See [[https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html Pytorch Python Embedding]]
  */
// format: on
final class Embedding[ParamType <: FloatNN | ComplexNN: Default](
    val num_embeddings: Int,
    val embedding_dim: Int,
    val padding_idx: Option[Int] | Int = None,
    val max_norm: Option[Float] | Float = None,
    val norm_type: Option[Float] | Float = Some(2.0f),
    val scale_grad_by_freq: Boolean | Option[Boolean] = false,
    val sparse: Boolean | Option[Boolean] = false
) extends HasParams[ParamType]
    with HasWeight[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options = new EmbeddingOptions(num_embeddings.toLong, embedding_dim.toLong)

  max_norm match {
    case m: Float         => options.max_norm().put(m.toDouble)
    case m: Option[Float] => if m.isDefined then options.max_norm().put(m.get.toDouble)
  }

  norm_type match {
    case n: Float         => options.norm_type().put(n.toDouble)
    case n: Option[Float] => if n.isDefined then options.norm_type().put(n.get.toDouble)
  }
  scale_grad_by_freq match {
    case s: Boolean         => options.scale_grad_by_freq().put(s)
    case s: Option[Boolean] => if s.isDefined then options.scale_grad_by_freq().put(s.get)
  }
  sparse match {
    case s: Boolean         => options.sparse().put(s)
    case s: Option[Boolean] => if s.isDefined then options.sparse().put(s.get)
  }

  padding_idx match {
    case p: Int         => options.padding_idx().put(p)
    case p: Option[Int] => if p.isDefined then options.padding_idx().put(p.get)
  }

  override val nativeModule: EmbeddingImpl = EmbeddingImpl(options)
  nativeModule.to(paramType.toScalarType, false)

  def reset(): Unit = nativeModule.reset()

  def reset_parameters(): Unit = nativeModule.reset_parameters()

  override def hasBias(): Boolean = false

  def weight: Tensor[ParamType] = fromNative(nativeModule.weight)

  def weight_=(w: Tensor[ParamType]): Tensor[ParamType] =
    nativeModule.weight(w.native)
    w

  def apply(indices: Tensor[ParamType]): Tensor[ParamType] = fromNative(
    nativeModule.forward(indices.to(torch.int64).native)
  )

  def apply(
      indices: Tensor[Int64] | Tensor[Int32],
      weight: Option[Tensor[ParamType]] = None
  ): Tensor[ParamType] = {
    indices.dtype match
      case torch.int64 => fromNative(nativeModule.forward(indices.native))
      case torch.int32 => fromNative(nativeModule.forward(indices.to(torch.int64).native))
  }
  def forward(
      indices: Tensor[Int64] | Tensor[Int32],
      weight: Option[Tensor[ParamType]] = None
  ): Tensor[ParamType] = {
    indices.dtype match
      case torch.int64 => fromNative(nativeModule.forward(indices.native))
      case torch.int32 => fromNative(nativeModule.forward(indices.to(torch.int64).native))
  }
  def forward(indices: Tensor[ParamType]): Tensor[ParamType] = fromNative(
    nativeModule.forward(indices.to(torch.int64).native)
  )

  override def toString(): String =
    val numEmbed = s"numEmbeddings=$num_embeddings"
    val dim = s"embeddingDim=$embedding_dim"
    val padding = s"paddingIdx=$padding_idx"
    val maxN = s"maxNorm=$max_norm"
    val normT = s"normType=$norm_type"
    val scale = s"scaleGradByFreq=$scale_grad_by_freq"
    val s = s"sparse=$sparse"
    s"${getClass().getSimpleName()}($numEmbed, $dim, $padding, $maxN, $normT, $scale, $s )"

object Embedding:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      num_embeddings: Int,
      embedding_dim: Int,
      padding_idx: Option[Int] | Int = None,
      max_norm: Option[Float] | Float = None,
      norm_type: Option[Float] | Float = Some(2.0f),
      scale_grad_by_freq: Boolean | Option[Boolean] = false,
      sparse: Boolean = false
  ): Embedding[ParamType] = new Embedding(
    num_embeddings,
    embedding_dim,
    padding_idx,
    max_norm,
    norm_type,
    scale_grad_by_freq,
    sparse
  )

  def from_pretrained[ParamType <: FloatNN | ComplexNN: Default](
      embeddings: Tensor[ParamType],
      freeze: Boolean = true,
      padding_idx: Option[Int] | Int = None,
      max_norm: Option[Float] | Float = None,
      norm_type: Option[Float] | Float = Some(2.0f),
      scale_grad_by_freq: Boolean | Option[Boolean] = false,
      sparse: Boolean | Option[Boolean] = false,
      include_last_offset: Boolean | Option[Boolean] = false
  ): Embedding[ParamType] = {
    require(embeddings.shape.length >= 2, "embeddings weight shape must have 2 dimension")
    val shape = embeddings.shape
    val num_embeddings = shape(0)
    val embedding_dim = shape(1)
    val embeddingModel: Embedding[ParamType] = new Embedding(
      num_embeddings,
      embedding_dim,
      padding_idx,
      max_norm,
      norm_type,
      scale_grad_by_freq,
      sparse
    )
    embeddingModel.weight_=(embeddings)
    if freeze then embeddingModel.weight.requires_grad = false
    else embeddingModel.weight.requires_grad = true
    embeddingModel
  }

//embeddings (Tensor) – FloatTensor containing weights for the Embedding. First dimension is being passed to Embedding as num_embeddings, second as embedding_dim.
//
//freeze (bool, optional) – If True, the tensor does not get updated in the learning process. Equivalent to embedding.weight.requires_grad = False. Default: True
//
//padding_idx (int, optional) – If specified, the entries at padding_idx do not contribute to the gradient; therefore, the embedding vector at padding_idx is not updated during training, i.e. it remains as a fixed “pad”.
//
//max_norm (float, optional) – See module initialization documentation.
//
//norm_type (float, optional) – See module initialization documentation. Default 2.
//
//scale_grad_by_freq (bool, optional) – See module initialization documentation. Default False.
//
//sparse (bool, optional) – See module initialization documentation.
//classmethod from_pretrained(embeddings, freeze=True,
// padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)

//    @classmethod
//    def from_pretrained(
//        cls,
//        embeddings,
//        freeze=True,
//        padding_idx=None,
//        max_norm=None,
//        norm_type=2.0,
//        scale_grad_by_freq=False,
//        sparse=False,
//  def apply(indices: Tensor[Int64],weight: Option[Tensor[ParamType]] = None): Tensor[ParamType] = fromNative(
//    nativeModule.forward(indices.native)
//  )

//case input: Tensor[Int32]
//  def apply(t: Tensor[ParamType]): Tensor[ParamType] = fromNative(nativeModule.forward(t.native))

//if paddingIdx.isDefined then options.padding_idx().put(toNative(paddingIdx.get))
//if maxNorm.isDefined then options.max_norm().put(maxNorm.get.toDouble)
//if normType.isDefined then options.norm_type().put(normType.get.toDouble)
//options.scale_grad_by_freq().put(scaleGradByFreq)
//options.sparse().put(sparse)
//paddingIdx.foreach(p => options.padding_idx().put(toNative(p)))
//maxNorm.foreach(m => options.max_norm().put(m))
//normType.foreach(n => options.norm_type().put(n))
