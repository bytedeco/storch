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
package recurrent

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{GRUCellImpl, GRUCellOptions, TensorVector}
import torch.internal.NativeConverters.{fromNative, toNative}

import torch.{Default, FloatNN, ComplexNN, Tensor}

/** Applies a 2D convolution over an input signal composed of several input planes. long input_size,
  * \@Cast("int64_t") long hidden_size
  * @group nn_conv
  */
final class GRUCell[ParamType <: FloatNN | ComplexNN: Default](
    val input_size: Int,
    val hidden_size: Int,
    val bias: Boolean = true
) extends HasParams[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options = new GRUCellOptions(input_size.toLong, hidden_size.toLong)
  options.bias().put(bias)

  override private[torch] val nativeModule: GRUCellImpl = GRUCellImpl(options)
  nativeModule.to(paramType.toScalarType, false)

  def weight_ih(weight: Tensor[ParamType]) = nativeModule.weight_ih(weight.native)

  def weight_hh(weight: Tensor[ParamType]) = nativeModule.weight_hh(weight.native)

  def bias_ih(bias: Tensor[ParamType]) = nativeModule.bias_ih(bias.native)

  def bias_hh(bias: Tensor[ParamType]) = nativeModule.bias_hh(bias.native)

  def weight_ih(): Tensor[ParamType] = fromNative(nativeModule.weight_ih())

  def weight_hh(): Tensor[ParamType] = fromNative(nativeModule.weight_hh())

  def bias_ih(): Tensor[ParamType] = fromNative(nativeModule.bias_ih())

  def bias_hh(): Tensor[ParamType] = fromNative(nativeModule.bias_hh())

  def reset(): Unit = nativeModule.reset()

  def reset_parameters(): Unit = nativeModule.reset_parameters()

  def apply(input: Tensor[ParamType], hx: Tensor[ParamType]): Tensor[ParamType] = {
    val fore = nativeModule.forward(input.native, hx.native)
    fromNative(fore)
  }
  def apply(input: Tensor[ParamType], hx: Option[Tensor[ParamType]] = None): Tensor[ParamType] = {
    val fore =
      if hx.isDefined then nativeModule.forward(input.native, hx.get.native)
      else nativeModule.forward(input.native)
    fromNative(fore)
  }

  def forward(input: Tensor[ParamType], hx: Option[Tensor[ParamType]] = None): Tensor[ParamType] = {
    val fore =
      if hx.isDefined then nativeModule.forward(input.native, hx.get.native)
      else nativeModule.forward(input.native)
    fromNative(fore)
  }
  def forward(input: Tensor[ParamType], hx: Tensor[ParamType]): Tensor[ParamType] = {
    val fore = nativeModule.forward(input.native, hx.native)
    fromNative(fore)
  }

  def weight: TensorVector = TensorVector(
    nativeModule.weight_hh(),
    nativeModule.weight_ih() // .all_weights()
  ) // .weight_hh()

  override def hasBias(): Boolean = options.bias().get()

  override def toString =
    s"${getClass.getSimpleName}(inputSize=$input_size, hiddenSize=$hidden_size bias=$bias)"

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???

object GRUCell:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      input_size: Int,
      hidden_size: Int,
      bias: Boolean = true
  ): GRUCell[ParamType] = new GRUCell(input_size, hidden_size, bias)

//  def weight: TensorVector = fromNative(nativeModule.all_weights())
