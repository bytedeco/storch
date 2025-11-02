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
package attention
import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.javacpp.{LongPointer, DoublePointer}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{MultiheadAttentionImpl, MultiheadAttentionOptions}
import torch.internal.NativeConverters.{fromNative, toNative}

/** Applies a 2D convolution over an input signal composed of several input planes. long input_size,
  * \@Cast("int64_t") long hidden_size T_TensorT_TensorTensor_T_T, T_TensorTensor_T,
  * T_TensorTensor_TOptional,
  *
  * @group nn_conv
  */
final class MultiheadAttention[ParamType <: FloatNN | ComplexNN: Default](
    val embed_dim: Int,
    val num_heads: Int,
    val dropout: Float | Double = 0.0f,
    val bias: Boolean = true,
    val add_bias_kv: Boolean = false,
    val add_zero_attn: Boolean = false,
    val kdim: Int | Option[Int] = None,
    val vdim: Int | Option[Int] = None,
    val batch_first: Boolean = false
) extends HasParams[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options = new MultiheadAttentionOptions(embed_dim.toLong, num_heads.toLong)
  options.embed_dim().put(LongPointer(1).put(embed_dim.toLong))
  options.num_heads().put(LongPointer(1).put(num_heads.toLong))
  dropout match {
    case d: Double => options.dropout().put(DoublePointer(1).put(d))
    case d: Float  => options.dropout().put(DoublePointer(1).put(d.toDouble))
  }

  options.bias().put(bias)
  options.add_bias_kv().put(add_bias_kv)
  options.add_zero_attn().put(add_zero_attn)

  kdim match {
    case k: Int => options.kdim().put(k.toLong)
    case k: Option[Int] =>
      if k.isDefined then options.kdim().put(LongPointer(1).put(k.get.toLong))
      else options.kdim().put(LongPointer(1).put(embed_dim.toLong))
  }
  vdim match {
    case v: Int => options.vdim().put(v.toLong)
    case v: Option[Int] =>
      if v.isDefined then options.vdim().put(LongPointer(1).put(v.get.toLong))
      else options.vdim().put(LongPointer(1).put(embed_dim.toLong))
  }

  override private[torch] val nativeModule: MultiheadAttentionImpl = MultiheadAttentionImpl(options)
  nativeModule.to(paramType.toScalarType, false)

  def apply(
      query: Tensor[ParamType],
      key: Tensor[ParamType],
      value: Tensor[ParamType]
  ): Tuple2[Tensor[ParamType], Tensor[ParamType]] = {
    val fore = nativeModule.forward(query.native, key.native, value.native)
    (fromNative(fore.get0()), fromNative(fore.get1()))
  }
  def forward(
      query: Tensor[ParamType],
      key: Tensor[ParamType],
      value: Tensor[ParamType]
  ): Tuple2[Tensor[ParamType], Tensor[ParamType]] = {
    val fore = nativeModule.forward(query.native, key.native, value.native)
    (fromNative(fore.get0()), fromNative(fore.get1()))
  }

  def apply(
      query: Tensor[ParamType],
      key: Tensor[ParamType],
      value: Tensor[ParamType],
      key_padding_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None,
      need_weights: Boolean = true,
      attn_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None,
      average_attn_weights: Boolean = true
  ): Tuple2[Tensor[ParamType], Tensor[ParamType]] = {
    this.forward(
      query,
      key,
      value,
      key_padding_mask,
      need_weights,
      attn_mask,
      average_attn_weights
    )
  }
  def forward(
      query: Tensor[ParamType],
      key: Tensor[ParamType],
      value: Tensor[ParamType],
      key_padding_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None,
      need_weights: Boolean = true,
      attn_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None,
      average_attn_weights: Boolean = true
  ): Tuple2[Tensor[ParamType], Tensor[ParamType]] = {
    val kpm = key_padding_mask match {
      case k: Tensor[ParamType]         => k.native
      case k: Option[Tensor[ParamType]] => if k.isDefined then k.get.native else torchNative.empty()
    }
    val am = attn_mask match {
      case a: Tensor[ParamType]         => a.native
      case a: Option[Tensor[ParamType]] => if a.isDefined then a.get.native else torchNative.empty()
    }
    val fore = nativeModule.forward(
      query.native,
      key.native,
      value.native,
      kpm,
      need_weights,
      am,
      average_attn_weights
    )
    (fromNative(fore.get0()), fromNative(fore.get1()))
  }

  def reset(): Unit = nativeModule.reset()

  def reset_parameters(): Unit = nativeModule._reset_parameters()

  def qkv_same_embed_dim(): Boolean = nativeModule._qkv_same_embed_dim()

  def in_proj_weight(weight: Tensor[ParamType]): MultiheadAttentionImpl =
    nativeModule.in_proj_weight(weight.native)

  def in_proj_bias(bias: Tensor[ParamType]): MultiheadAttentionImpl =
    nativeModule.in_proj_bias(bias.native)

  def bias_k(bias: Tensor[ParamType]): MultiheadAttentionImpl = nativeModule.bias_k(bias.native)

  def bias_v(bias: Tensor[ParamType]): MultiheadAttentionImpl = nativeModule.bias_v(bias.native)

  def q_proj_weight(weight: Tensor[ParamType]): MultiheadAttentionImpl =
    nativeModule.q_proj_weight(weight.native)

  def k_proj_weight(weight: Tensor[ParamType]): MultiheadAttentionImpl =
    nativeModule.k_proj_weight(weight.native)

  def v_proj_weight(weight: Tensor[ParamType]): MultiheadAttentionImpl =
    nativeModule.v_proj_weight(weight.native)

  def in_proj_weight(): Tensor[ParamType] = fromNative(nativeModule.in_proj_weight())

  def in_proj_bias(): Tensor[ParamType] = fromNative(nativeModule.in_proj_bias())

  def bias_k(): Tensor[ParamType] = fromNative(nativeModule.bias_k())

  def bias_v(): Tensor[ParamType] = fromNative(nativeModule.bias_v())

  def q_proj_weight(): Tensor[ParamType] = fromNative(nativeModule.q_proj_weight())

  def k_proj_weight(): Tensor[ParamType] = fromNative(nativeModule.k_proj_weight())

  def v_proj_weight(): Tensor[ParamType] = fromNative(nativeModule.v_proj_weight())

  def head_dim(): Long = nativeModule.head_dim()

  override def hasBias(): Boolean = options.bias().get()

  override def toString(): String =
    s"${getClass().getSimpleName()}(embedDim=$embed_dim numHeads=$num_heads dropout=$dropout bias=$bias addBiasKV=$add_bias_kv addZeroAttn=$add_zero_attn kdim=$kdim vdim=$vdim)"

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???

object MultiheadAttention:

  def apply[ParamType <: FloatNN | ComplexNN: Default](
      embed_dim: Int,
      num_heads: Int,
      dropout: Float | Double = 0.0f,
      bias: Boolean = true,
      add_bias_kv: Boolean = false,
      add_zero_attn: Boolean = false,
      kdim: Int | Option[Int] = None,
      vdim: Int | Option[Int] = None,
      batch_first: Boolean = false
  ): MultiheadAttention[ParamType] = new MultiheadAttention[ParamType](
    embed_dim,
    num_heads,
    dropout,
    bias,
    add_bias_kv,
    add_zero_attn,
    kdim,
    vdim,
    batch_first
  )
  enum TransformerActivation:
    case kReLU, kGELU, TensorMapper
