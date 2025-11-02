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

import org.bytedeco.javacpp.{LongPointer}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  kReLU,
  kGELU,
  AnyModule,
  TransformerDecoderImpl,
  TransformerDecoderOptions
}
import torch.internal.NativeConverters.{fromNative, toNative}
//import torch.nn.modules.attention.TransformerDecoderLayer

import torch.nn.modules.attention.Transformer.TransformerActivation

/** Applies a 2D convolution over an input signal composed of several input planes. long input_size,
  * \@Cast("int64_t") long hidden_size
  * @group nn_conv
  */
final class TransformerDecoder[ParamType <: FloatNN | ComplexNN: Default](
    val decoder_layer: TransformerDecoderLayer[ParamType],
    val num_layers: Int,
    val norm: Option[AnyModule] = None,
    val layer_norm_eps: Float = 1e-5,
    val batch_first: Boolean = false,
    val norm_first: Boolean = false,
    val activation: TransformerActivation | String = TransformerActivation.kReLU,
    val bias: Boolean = true
) extends HasParams[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  protected val layerOptions = decoder_layer.options

  override def toString =
    s"${getClass.getSimpleName}(numLayers=$num_layers, bias=$bias activation=${activation.toString} norm=$norm )"

  activation match {
    case TransformerActivation.kReLU | "relu" | "Relu" | "ReLU" | "RELU" | "ReLu" =>
      layerOptions.activation().put(new kReLU)
    case TransformerActivation.kGELU | "gelu" | "Gelu" | "GeLU" | "GELU" | "GeLu" =>
      layerOptions.activation().put(new kGELU)
  }

  private val options = new TransformerDecoderOptions(layerOptions, num_layers.toLong)

  options.num_layers().put(LongPointer(1).put(num_layers.toLong))
  if (norm.isDefined) options.norm().put(norm.get)

  override private[torch] val nativeModule: TransformerDecoderImpl = TransformerDecoderImpl(options)
  nativeModule.to(paramType.toScalarType, false)

  def reset(): Unit = nativeModule.reset()

  def reset_parameters(): Unit = nativeModule.reset_parameters()

  override def hasBias(): Boolean = false // options.bias().get()

  def norm_raw() = nativeModule.norm()

  def apply(
      tgt: Tensor[ParamType],
      memory: Tensor[ParamType],
      tgt_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None,
      memory_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None,
      tgt_key_padding_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None,
      memory_key_padding_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None
  ): Tensor[ParamType] = {
    this.forward(
      tgt,
      memory,
      tgt_mask,
      memory_mask,
      tgt_key_padding_mask,
      memory_key_padding_mask
    )
  }
  def forward(
      tgt: Tensor[ParamType],
      memory: Tensor[ParamType],
      tgt_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None,
      memory_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None,
      tgt_key_padding_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None,
      memory_key_padding_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None
  ): Tensor[ParamType] = {
    val tgtMask = tgt_mask match {
      case a: Tensor[ParamType]         => a.native
      case a: Option[Tensor[ParamType]] => if a.isDefined then a.get.native else torchNative.empty()
    }
    val memoryMask = memory_mask match {
      case k: Tensor[ParamType]         => k.native
      case k: Option[Tensor[ParamType]] => if k.isDefined then k.get.native else torchNative.empty()
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
      if (tgtMask.equals(torchNative.empty()))
        nativeModule.forward(
          tgt.native,
          memory.native
        )
      else
        nativeModule.forward(
          tgt.native,
          memory.native,
          tgtMask,
          memoryMask,
          tgtKPM,
          memoryKPM
        )
    fromNative(fore)
  }

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???

object TransformerDecoder:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      decoder_layer: TransformerDecoderLayer[ParamType],
      num_layers: Int,
      norm: Option[AnyModule] = None,
      layer_norm_eps: Float = 1e-5,
      batch_first: Boolean = false,
      norm_first: Boolean = false,
      activation: TransformerActivation | String = TransformerActivation.kReLU,
      bias: Boolean = true
  ): TransformerDecoder[ParamType] =
    new TransformerDecoder(
      decoder_layer,
      num_layers,
      norm,
      layer_norm_eps,
      batch_first,
      norm_first,
      activation,
      bias
    )
  def makeInstance[ParamType <: FloatNN | ComplexNN: Default](
      num_layers: Int,
      d_model: Int,
      n_head: Int,
      dim_feedforward: Int = 2048,
      dropout: Float = 0.1,
      layer_norm_eps: Float = 1e-5,
      batch_first: Boolean = false,
      norm_first: Boolean = false,
      activation: TransformerActivation = TransformerActivation.kReLU,
      norm: Option[AnyModule] = None,
      bias: Boolean = true
  ): TransformerDecoder[ParamType] = {
    val encoderLayer = new TransformerDecoderLayer[ParamType](
      d_model,
      n_head,
      dim_feedforward,
      dropout,
      activation,
      layer_norm_eps,
      batch_first,
      norm_first,
      bias
    )
    new TransformerDecoder[ParamType](
      encoderLayer,
      num_layers,
      norm,
      layer_norm_eps,
      batch_first,
      norm_first,
      activation,
      bias
    )
  }
