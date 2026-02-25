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
  TransformerEncoderImpl,
  TransformerEncoderOptions
}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn.modules.attention.TransformerEncoderLayer
import torch.nn.modules.attention.Transformer.TransformerActivation

/** Applies a 2D convolution over an input signal composed of several input planes. long input_size,
  * \@Cast("int64_t") long hidden_size T_TensorT_TensorTensor_T_T, T_TensorTensor_T,
  * T_TensorTensor_TOptional, TensorVector,
  *
  * @group nn_conv
  */
final class TransformerEncoder[ParamType <: FloatNN | ComplexNN: Default](
    val encoder_layer: TransformerEncoderLayer[ParamType],
    val num_layers: Int,
    val activation: TransformerActivation | String = TransformerActivation.kReLU,
    val norm: Option[AnyModule] = None,
    val enable_nested_tensor: Boolean = true,
    val layer_norm_eps: Float = 1e-5,
    val batch_first: Boolean = false,
    val norm_first: Boolean = false,
    val bias: Boolean = true
) extends HasParams[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  protected val layerOptions = encoder_layer.options

  activation match {
    case TransformerActivation.kReLU | "relu" | "Relu" | "ReLU" | "RELU" | "ReLu" =>
      layerOptions.activation().put(new kReLU)
    case TransformerActivation.kGELU | "gelu" | "Gelu" | "GeLU" | "GELU" | "GeLu" =>
      layerOptions.activation().put(new kGELU)
  }

  private val options = new TransformerEncoderOptions(layerOptions, num_layers.toLong)

  options.num_layers().put(LongPointer(1).put(num_layers.toLong))
  if (norm.isDefined) options.norm().put(norm.get)

  override private[torch] val nativeModule: TransformerEncoderImpl = TransformerEncoderImpl(options)
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

  def norm_raw() = nativeModule.norm()

  def reset(): Unit = nativeModule.reset()

  def reset_parameters(): Unit = nativeModule.reset_parameters()

  override def toString =
    s"${getClass.getSimpleName}(numLayers=$num_layers, bias=$bias activation=${activation.toString} norm=$norm   )"

  override def apply(src: Tensor[ParamType]): Tensor[ParamType] = fromNative(
    nativeModule.forward(src.native)
  )

object TransformerEncoder:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      encoder_layer: TransformerEncoderLayer[ParamType],
      num_layers: Int,
      activation: TransformerActivation = TransformerActivation.kReLU,
      norm: Option[AnyModule] = None,
      enable_nested_tensor: Boolean = true,
      layer_norm_eps: Float = 1e-5,
      batch_first: Boolean = false,
      norm_first: Boolean = false,
      bias: Boolean = true
  ): TransformerEncoder[ParamType] =
    new TransformerEncoder(
      encoder_layer,
      num_layers,
      activation,
      norm,
      enable_nested_tensor,
      layer_norm_eps,
      batch_first,
      norm_first,
      bias
    )

  def makeInstance[ParamType <: FloatNN | ComplexNN: Default](
      num_layers: Int,
      d_model: Int,
      n_head: Int,
      dim_feedforward: Int = 2048,
      dropout: Float = 0.1,
      activation: TransformerActivation | String = TransformerActivation.kReLU,
      norm: Option[AnyModule] = None,
      enable_nested_tensor: Boolean = true,
      layer_norm_eps: Float = 1e-5,
      batch_first: Boolean = false,
      norm_first: Boolean = false,
      bias: Boolean = true
  ): TransformerEncoder[ParamType] = new TransformerEncoder[ParamType](
    new TransformerEncoderLayer[ParamType](
      d_model,
      n_head,
      dim_feedforward,
      dropout,
      activation,
      layer_norm_eps,
      batch_first,
      norm_first,
      bias
    ),
    num_layers,
    activation,
    norm,
    enable_nested_tensor,
    layer_norm_eps,
    batch_first,
    norm_first,
    bias
  )
