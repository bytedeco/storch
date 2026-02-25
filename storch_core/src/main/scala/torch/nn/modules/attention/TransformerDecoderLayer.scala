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
import org.bytedeco.pytorch.{
  kReLU,
  kGELU,
  TransformerDecoderLayerImpl,
  TransformerDecoderLayerOptions
}
import torch.internal.NativeConverters.{fromNative, toNative}
//import torch.nn.modules.attention.Transformer.TransformerActivation

import torch.nn.modules.attention.Transformer.TransformerActivation

/** Applies a 2D convolution over an input signal composed of several input planes. long input_size,
  * \@Cast("int64_t") long hidden_size
  * @group nn_conv
  */
final class TransformerDecoderLayer[ParamType <: FloatNN | ComplexNN: Default](
    val d_model: Int,
    val nhead: Int,
    val dim_feedforward: Int = 2048,
    val dropout: Float | Double = 0.1,
    val activation: TransformerActivation | String = TransformerActivation.kReLU,
    val layer_norm_eps: Float = 1e-5,
    val batch_first: Boolean = false,
    val norm_first: Boolean = false,
    val bias: Boolean = true
) extends HasParams[ParamType]
    with TensorModule[ParamType]:
  override def toString =
    s"${getClass.getSimpleName}(dModel=$d_model, nHead=$nhead activation=${activation.toString} dimFeedforward=$dim_feedforward dropout=  $dropout bias=$bias)"

  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  val options = new TransformerDecoderLayerOptions(d_model.toLong, nhead.toLong)
  options.d_model().put(LongPointer(1).put(d_model.toLong))
  options.nhead().put(LongPointer(1).put(nhead.toLong))
  options.dim_feedforward().put(LongPointer(1).put(dim_feedforward.toLong))
  dropout match {
    case d: Double => options.dropout().put(DoublePointer(1).put(d))
    case d: Float  => options.dropout().put(DoublePointer(1).put(d.toDouble))
  }

  activation match {
    case TransformerActivation.kReLU | "relu" | "Relu" | "ReLU" | "RELU" | "ReLu" =>
      options.activation().put(new kReLU)
    case TransformerActivation.kGELU | "gelu" | "Gelu" | "GeLU" | "GELU" | "GeLu" =>
      options.activation().put(new kGELU)
  }

  override private[torch] val nativeModule: TransformerDecoderLayerImpl =
    TransformerDecoderLayerImpl(options)
  nativeModule.to(paramType.toScalarType, false)

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
      if (tgtMask.equals(torchNative.empty()) && memoryMask.equals(torchNative.empty()))
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

  override def hasBias(): Boolean = false // options.bias().get()

  def reset(): Unit = nativeModule.reset()

  def reset_parameters(): Unit = nativeModule.reset_parameters()

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???

object TransformerDecoderLayer:

  def apply[PT <: FloatNN | ComplexNN: Default](
      d_model: Int,
      nhead: Int,
      dim_feedforward: Int = 2048,
      dropout: Float | Double = 0.1,
      activation: TransformerActivation | String = TransformerActivation.kReLU,
      layer_norm_eps: Float = 1e-5,
      batch_first: Boolean = false,
      norm_first: Boolean = false,
      bias: Boolean = true
  ): TransformerDecoderLayer[PT] = new TransformerDecoderLayer[PT](
    d_model,
    nhead,
    dim_feedforward,
    dropout,
    activation,
    layer_norm_eps,
    batch_first,
    norm_first,
    bias
  )
