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

import org.bytedeco.javacpp.{LongPointer, DoublePointer, BoolPointer}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  kReLU,
  kGELU,
  TransformerEncoderLayerImpl,
  TransformerEncoderLayerOptions
}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn.modules.attention.Transformer.TransformerActivation

final class TransformerEncoderLayer[ParamType <: FloatNN | ComplexNN: Default](
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
    s"${getClass.getSimpleName}(dModel=$d_model, nHead=$nhead activation=${activation.toString} dimFeedforward=$dim_feedforward dropout= $dropout bias=$bias)"

  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  val options = new TransformerEncoderLayerOptions(d_model.toLong, nhead.toLong)
  options.d_model().put(d_model)
  options.nhead().put(nhead.toLong)
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

  override private[torch] val nativeModule: TransformerEncoderLayerImpl =
    TransformerEncoderLayerImpl(options)
  nativeModule.to(paramType.toScalarType, false)

  def apply(
      src: Tensor[ParamType],
      src_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None,
      src_key_padding_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None
  ): Tensor[ParamType] = {
    this.forward(src, src_mask, src_key_padding_mask)
  }
  def forward(
      src: Tensor[ParamType],
      src_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None,
      src_key_padding_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None
  ): Tensor[ParamType] = {
    val srcMask = src_mask match {
      case k: Tensor[ParamType]         => k.native
      case k: Option[Tensor[ParamType]] => if k.isDefined then k.get.native else torchNative.empty()
    }

    val srcKPM = src_key_padding_mask match {
      case k: Tensor[ParamType]         => k.native
      case k: Option[Tensor[ParamType]] => if k.isDefined then k.get.native else torchNative.empty()
    }
    val fore =
      if (srcMask.equals(torchNative.empty()) && srcKPM.equals(torchNative.empty()))
        nativeModule.forward(src.native)
      else nativeModule.forward(src.native, srcMask, srcKPM)
    fromNative(fore)
  }

  override def hasBias(): Boolean = false // options.bias().get()

  def reset(): Unit = nativeModule.reset()

  def reset_parameters(): Unit = nativeModule.reset_parameters()

  override def apply(src: Tensor[ParamType]): Tensor[ParamType] = fromNative(
    nativeModule.forward(src.native)
  )

object TransformerEncoderLayer:
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
  ): TransformerEncoderLayer[PT] = new TransformerEncoderLayer[PT](
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

//  options.d_model().put(LongPointer(1).put(dModel.toLong))
//  options.nhead().put(LongPointer(1).put(nHead.toLong))
