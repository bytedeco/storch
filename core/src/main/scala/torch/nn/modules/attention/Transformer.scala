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
import org.bytedeco.pytorch.{kReLU, kGELU, AnyModule, TransformerImpl, TransformerOptions}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn.modules.attention.Transformer.TransformerActivation

/** Applies a 2D convolution over an input signal composed of several input planes. long input_size,
  * \@Cast("int64_t") long hidden_size T_TensorT_TensorTensor_T_T, T_TensorTensor_T,
  * T_TensorTensor_TOptional, TensorVector,
  *
  * @group nn_conv custom_encoder ,custom_decoder,
  */
final class Transformer[ParamType <: FloatNN | ComplexNN: Default](
    val d_model: Int = 512,
    val nhead: Int = 8,
    val num_encoder_layers: Int = 6,
    val num_decoder_layers: Int = 6,
    val dim_feedforward: Int = 2048,
    val dropout: Float | Double = 0.1,
    val activation: TransformerActivation | String = TransformerActivation.kReLU,
    val custom_encoder: Option[AnyModule] = None,
    val custom_decoder: Option[AnyModule] = None,
    val layer_norm_eps: Float = 1e-05,
    val batch_first: Boolean = false,
    val norm_first: Boolean = false,
    val bias: Boolean = true
) extends HasParams[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options = new TransformerOptions(
    d_model.toLong,
    nhead.toLong,
    num_encoder_layers.toLong,
    num_decoder_layers.toLong
  )
  options.num_encoder_layers().put(LongPointer(1).put(num_encoder_layers.toLong))
  options.num_decoder_layers().put(LongPointer(1).put(num_decoder_layers.toLong))
  options.dim_feedforward().put(LongPointer(1).put(dim_feedforward.toLong))
  dropout match {
    case d: Double => options.dropout().put(DoublePointer(1).put(d))
    case d: Float  => options.dropout().put(DoublePointer(1).put(d.toDouble))
  }
  options.nhead().put(LongPointer(1).put(nhead.toLong))
  options.d_model().put(LongPointer(1).put(d_model.toLong))

  if (custom_encoder.isDefined) options.custom_encoder().put(custom_encoder.get)
  if (custom_decoder.isDefined) options.custom_decoder().put(custom_decoder.get)

  activation match {
    case TransformerActivation.kReLU | "relu" | "Relu" | "RELU" | "ReLu" | "ReLU" =>
      options.activation().put(new kReLU)
    case TransformerActivation.kGELU | "gelu" | "Gelu" | "GELU" | "GeLu" | "GeLU" =>
      options.activation().put(new kGELU)

  }

  override private[torch] val nativeModule: TransformerImpl = TransformerImpl(options)
  nativeModule.to(paramType.toScalarType, false)
  def reset(): Unit = nativeModule.reset()

  def reset_parameters(): Unit = nativeModule.reset_parameters()

  def generate_square_subsequent_mask(sz: Int): Tensor[ParamType] = fromNative(
    TransformerImpl.generate_square_subsequent_mask(sz.toLong)
  ) // generate_square_subsequent_mask

  def encoder(encoderModule: AnyModule) = nativeModule.encoder(encoderModule)

  def decoder(decoderModule: AnyModule) = nativeModule.decoder(decoderModule)

  def apply(
      src: Tensor[ParamType],
      tgt: Tensor[ParamType],
      src_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None,
      tgt_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None,
      memory_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None,
      src_key_padding_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None,
      tgt_key_padding_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None,
      memory_key_padding_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None
  ): Tensor[ParamType] = {
    this.forward(
      src,
      tgt,
      src_mask,
      tgt_mask,
      memory_mask,
      src_key_padding_mask,
      tgt_key_padding_mask,
      memory_key_padding_mask
    )
  }
  def forward(
      src: Tensor[ParamType],
      tgt: Tensor[ParamType],
      src_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None,
      tgt_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None,
      memory_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None,
      src_key_padding_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None,
      tgt_key_padding_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None,
      memory_key_padding_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None
  ): Tensor[ParamType] = {
    val srcMask = src_mask match {
      case k: Tensor[ParamType]         => k.native
      case k: Option[Tensor[ParamType]] => if k.isDefined then k.get.native else torchNative.empty()
    }
    val tgtMask = tgt_mask match {
      case a: Tensor[ParamType]         => a.native
      case a: Option[Tensor[ParamType]] => if a.isDefined then a.get.native else torchNative.empty()
    }
    val memoryMask = memory_mask match {
      case k: Tensor[ParamType]         => k.native
      case k: Option[Tensor[ParamType]] => if k.isDefined then k.get.native else torchNative.empty()
    }
    val srcKPM = src_key_padding_mask match {
      case a: Tensor[ParamType]         => a.native
      case a: Option[Tensor[ParamType]] => if a.isDefined then a.get.native else torchNative.empty()
    }
    val tgtKPM = tgt_key_padding_mask match {
      case k: Tensor[ParamType]         => k.native
      case k: Option[Tensor[ParamType]] => if k.isDefined then k.get.native else torchNative.empty()
    }
    val memoryKPM = memory_key_padding_mask match {
      case a: Tensor[ParamType]         => a.native
      case a: Option[Tensor[ParamType]] => if a.isDefined then a.get.native else torchNative.empty()
    }
    val fore =
      if (srcMask.equals(torchNative.empty()))
        nativeModule.forward(
          src.native,
          tgt.native
        )
      else
        nativeModule.forward(
          src.native,
          tgt.native,
          srcMask,
          tgtMask,
          memoryMask,
          srcKPM,
          tgtKPM,
          memoryKPM
        )
    fromNative(fore)
  }

  override def hasBias(): Boolean = false

  override def toString =
    s"${getClass.getSimpleName}(dimFeedforward=$dim_feedforward, bias=$bias d_model=$d_model nhead=$nhead numEncoderLayers= ${num_encoder_layers} numDecoderLayers= ${num_decoder_layers} dropout=$dropout customEncoder=$custom_encoder customDecoder=$custom_decoder activation=$activation  )"

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???

// layer_norm_eps=1e-05, batch_first=False, norm_first=False,
object Transformer:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      d_model: Int = 512,
      nhead: Int = 8,
      num_encoder_layers: Int = 6,
      num_decoder_layers: Int = 6,
      dim_feedforward: Int = 2048,
      dropout: Float | Double = 0.1,
      activation: TransformerActivation | String = TransformerActivation.kReLU,
      custom_encoder: Option[AnyModule] = None,
      custom_decoder: Option[AnyModule] = None,
      layer_norm_eps: Float = 1e-05,
      batch_first: Boolean = false,
      norm_first: Boolean = false,
      bias: Boolean = true
  ): Transformer[ParamType] = new Transformer[ParamType](
    d_model,
    nhead,
    num_encoder_layers,
    num_decoder_layers,
    dim_feedforward,
    dropout,
    activation,
    custom_encoder,
    custom_decoder,
    layer_norm_eps,
    batch_first,
    norm_first,
    bias
  )
  enum TransformerActivation:
    case kReLU, kGELU, TensorMapper
