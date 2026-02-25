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
import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.pytorch
import org.bytedeco.pytorch.EmbeddingBagImpl
import org.bytedeco.pytorch.{kSum, kMean, kMax}
import org.bytedeco.pytorch.EmbeddingBagOptions
import torch.nn.modules.sparse.EmbeddingBag.EmbeddingBagMode
import torch.internal.NativeConverters.{fromNative, toNative}
import org.bytedeco.pytorch.global.torch.ScalarType
// format: off
/** A simple lookup table that stores embeddings of a fixed dictionary and size.
 *
 * This module is often used to store word embeddings and retrieve them using indices. The input to
 * the module is a list of indices, and the output is the corresponding word embeddings.
 *
 * @group nn_sparse
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
 * @see See [[https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_embedding.html#class-embedding Pytorch C++ Embedding]]
 * @see See [[https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html Pytorch Python Embedding]]
 * @native private def allocate(@Cast(Array("int64_t"))  num_embeddings: Long, @Cast(Array("int64_t"))  embedding_dim: Long): Unit
 *        }
 *        torch.nn.EmbeddingBag(num_embeddings, embedding_dim, max_norm=None,
 *        norm_type=2.0, scale_grad_by_freq=False, mode='mean', sparse=False,
 *        _weight=None,
 *        include_last_offset=False,
 *        padding_idx=None, device=None, dtype=None)
 */
// format: on
final class EmbeddingBag[ParamType <: FloatNN | ComplexNN: Default](
    val num_embeddings: Int,
    val embedding_dim: Int,
    val max_norm: Option[Float] | Float = None,
    val norm_type: Option[Float] | Float = Some(2.0f),
    val scale_grad_by_freq: Boolean | Option[Boolean] = false,
    val mode: EmbeddingBagMode | String = EmbeddingBagMode.kMean,
    val sparse: Boolean | Option[Boolean] = false,
    val include_last_offset: Boolean | Option[Boolean] = false,
    val padding_idx: Option[Int] | Int = None,
    val need_weight: Option[Tensor[ParamType]] = None
) extends HasParams[ParamType]
    with HasWeight[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options = new EmbeddingBagOptions(num_embeddings.toLong, embedding_dim.toLong)

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
  include_last_offset match {
    case i: Boolean         => options.include_last_offset().put(i)
    case i: Option[Boolean] => if i.isDefined then options.include_last_offset().put(i.get)
  }

  padding_idx match {
    case p: Int         => options.padding_idx().put(p)
    case p: Option[Int] => if p.isDefined then options.padding_idx().put(p.get)
  }

  if need_weight.isDefined then options._weight().put(need_weight.get.native)

  mode match
    case EmbeddingBagMode.kMean | "mean" | "Mean" | "MEAN" => options.mode().put(new kMean)
    case EmbeddingBagMode.kMax | "max" | "Max" | "MAX"     => options.mode().put(new kMax)
    case EmbeddingBagMode.kSum | "sum" | "Sum" | "SUM"     => options.mode().put(new kSum)

  override val nativeModule: EmbeddingBagImpl = EmbeddingBagImpl(options)
  nativeModule.to(paramType.toScalarType, false)

  override def hasBias(): Boolean = false

  def weight: Tensor[ParamType] = fromNative(nativeModule.weight)

  def weight_=(w: Tensor[ParamType]): Tensor[ParamType] =
    nativeModule.weight(w.native)
    w

  def reset(): Unit = nativeModule.reset()

  def reset_parameters(): Unit = nativeModule.reset_parameters()

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = fromNative(
    nativeModule.forward(input.native.to(ScalarType.Long))
  )
  def forward(input: Tensor[ParamType]): Tensor[ParamType] = fromNative(
    nativeModule.forward(input.native.to(ScalarType.Long))
  )

  def apply(
      indices: Tensor[Int64] | Tensor[Int32],
      raw_unused_false: Boolean
  ): Tensor[ParamType] = {
    indices.dtype match
      case torch.int64 => fromNative(nativeModule.forward(indices.native.to(ScalarType.Long)))
      case torch.int32 => fromNative(nativeModule.forward(indices.native.to(ScalarType.Long)))
  }

  def forward(
      indices: Tensor[Int64] | Tensor[Int32],
      raw_unused_false: Boolean
  ): Tensor[ParamType] = {
    indices.dtype match
      case torch.int64 => fromNative(nativeModule.forward(indices.native.to(ScalarType.Long)))
      case torch.int32 => fromNative(nativeModule.forward(indices.native.to(ScalarType.Long)))
  }

  def apply(
      input: Tensor[ParamType],
      offsets: Tensor[Int64] | Tensor[Int32],
      size: Seq[Int]
  ): Tensor[ParamType] = {
    fromNative(
      nativeModule.forward(
        input.native.to(ScalarType.Long),
        offsets.native.to(ScalarType.Long),
        torch.empty(size, dtype = float32, requires_grad = true).native.to(ScalarType.Float)
      )
    )
  }

  def forward(
      input: Tensor[ParamType],
      offsets: Tensor[Int64] | Tensor[Int32],
      size: Seq[Int]
  ): Tensor[ParamType] = {
    fromNative(
      nativeModule.forward(
        input.native.to(ScalarType.Long),
        offsets.native.to(ScalarType.Long),
        torch.empty(size, dtype = float32, requires_grad = true).native.to(ScalarType.Float)
      )
    )
  }

  def forward(
      input: Tensor[ParamType],
      offsets: Tensor[Int64] | Tensor[Int32] | Option[Tensor[Int64] | Tensor[Int32]] = None,
      per_sample_weights: Tensor[ParamType] | Option[Tensor[ParamType]] = None
  ): Tensor[ParamType] = {
    val offsetsNative = offsets match {
      case k: Tensor[Int32] => k.native.to(ScalarType.Long)
      case k: Tensor[Int64] => k.native.to(ScalarType.Long)
      case k: Option[Tensor[Int64] | Tensor[Int32]] =>
        if k.isDefined then k.get.native.to(ScalarType.Long)
        else torchNative.empty().to(ScalarType.Long)
    }
    val perSampleWeights = per_sample_weights match {
      case k: Tensor[ParamType] => k.native.to(ScalarType.Float)
      case k: Option[Tensor[ParamType]] =>
        if k.isDefined then k.get.native.to(ScalarType.Float)
        else torchNative.empty().to(ScalarType.Float)
    }
    fromNative(
      nativeModule.forward(
        input.native.to(ScalarType.Long),
        offsetsNative,
        perSampleWeights
      )
    )
  }

  def apply(
      input: Tensor[ParamType],
      offsets: Tensor[Int64] | Tensor[Int32] | Option[Tensor[Int64] | Tensor[Int32]] = None,
      per_sample_weights: Tensor[ParamType] | Option[Tensor[ParamType]] = None
  ): Tensor[ParamType] = {
    this.forward(
      input,
      offsets,
      per_sample_weights
    )

  }

  override def toString(): String =
    val numEmbed = s"numEmbeddings=$num_embeddings"
    val dim = s"embeddingDim=$embedding_dim"
    val padding = s"paddingIdx=$padding_idx"
    val maxN = s"maxNorm=$max_norm"
    val normT = s"normType=$norm_type"
    val scale = s"scaleGradByFreq=$scale_grad_by_freq"
    val s = s"sparse=$sparse"
    s"${getClass().getSimpleName()}($numEmbed, $dim, $padding, $maxN, $normT, $scale, $s )"

object EmbeddingBag:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      num_embeddings: Int,
      embedding_dim: Int,
      max_norm: Option[Float] | Float = None,
      norm_type: Option[Float] | Float = Some(2.0f),
      scale_grad_by_freq: Boolean | Option[Boolean] = false,
      mode: EmbeddingBagMode | String = EmbeddingBagMode.kMean,
      sparse: Boolean | Option[Boolean] = false,
      include_last_offset: Boolean | Option[Boolean] = false,
      padding_idx: Option[Int] | Int = None,
      need_weight: Option[Tensor[ParamType]] = None
  ): EmbeddingBag[ParamType] = new EmbeddingBag(
    num_embeddings,
    embedding_dim,
    max_norm,
    norm_type,
    scale_grad_by_freq,
    mode,
    sparse,
    include_last_offset,
    padding_idx,
    need_weight
  )

  def from_pretrained[ParamType <: FloatNN | ComplexNN: Default](
      embeddings: Tensor[ParamType],
      freeze: Boolean = true,
      max_norm: Option[Float] | Float = None,
      norm_type: Option[Float] | Float = Some(2.0f),
      scale_grad_by_freq: Boolean | Option[Boolean] = false,
      mode: EmbeddingBagMode | String = EmbeddingBagMode.kMean,
      sparse: Boolean | Option[Boolean] = false,
      include_last_offset: Boolean | Option[Boolean] = false,
      padding_idx: Option[Int] | Int = None,
      need_weight: Option[Tensor[ParamType]] = None
  ): EmbeddingBag[ParamType] = {

    require(embeddings.shape.length >= 2, "embeddings weight shape must have 2 dimension")
    val shape = embeddings.shape
    val num_embeddings = shape(0)
    val embedding_dim = shape(1)
    val embeddingBagModel: EmbeddingBag[ParamType] = new EmbeddingBag(
      num_embeddings,
      embedding_dim,
      max_norm,
      norm_type,
      scale_grad_by_freq,
      mode,
      sparse,
      include_last_offset,
      padding_idx,
      need_weight
    )
    embeddingBagModel.weight_=(embeddings)
    if freeze then embeddingBagModel.weight.requires_grad = false
    else embeddingBagModel.weight.requires_grad = true
    embeddingBagModel
  }

  enum EmbeddingBagMode:
    case kSum, kMean, kMax

//embeddings (Tensor) – FloatTensor containing weights for the EmbeddingBag. First dimension is being passed to EmbeddingBag as ‘num_embeddings’, second as ‘embedding_dim’.
//
//freeze (bool, optional) – If True, the tensor does not get updated in the learning process. Equivalent to embeddingbag.weight.requires_grad = False. Default: True
//
//max_norm (float, optional) – See module initialization documentation. Default: None
//
//norm_type (float, optional) – See module initialization documentation. Default 2.
//
//scale_grad_by_freq (bool, optional) – See module initialization documentation. Default False.
//
//mode (str, optional) – See module initialization documentation. Default: "mean"
//
//sparse (bool, optional) – See module initialization documentation. Default: False.
//
//include_last_offset (bool, optional) – See module initialization documentation. Default: False.
//
//padding_idx (int, optional) – See module initialization documentation. Default: None.
//classmethod from_pretrained(embeddings, freeze=True, max_norm=None, norm_type=2.0,
// scale_grad_by_freq=False, mode='mean', sparse=False,
// include_last_offset=False, padding_idx=None)

//    case input : Tensor[Int64] => fromNative(nativeModule.forward(indices.native.to(ScalarType.Long)))
//    case input : Tensor[Int32] => fromNative(nativeModule.forward(indices.native.to(ScalarType.Long)))

//  def apply(indices: Tensor[Int32 |Int64], weight: Option[String] = None,word:Option[String] = None,seq:Int =0): Tensor[ParamType] = fromNative(
//    nativeModule.forward(indices.native.to(ScalarType.Long))
//  )

//  paddingIdx.foreach(p => options.padding_idx().put(toNative(p)))
//  maxNorm.foreach(m => options.max_norm().put(m))
//  normType.foreach(n => options.norm_type().put(n))
//  options.mode().put(modeNative)

//if maxNorm.isDefined then options.max_norm().put(maxNorm.get.toDouble)
//if normType.isDefined then options.norm_type().put(normType.get.toDouble)
//options.scale_grad_by_freq().put(scaleGradByFreq)
//options.sparse().put(sparse)
//options.include_last_offset().put(includeLastOffset)
//if paddingIdx.isDefined then options.padding_idx().put(paddingIdx.get)
