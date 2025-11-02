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
import org.bytedeco.pytorch.{LSTMCellImpl, LSTMCellOptions, T_TensorTensor_T, TensorVector}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.{Default, FloatNN, ComplexNN, Tensor}

/** Applies a 2D convolution over an input signal composed of several input planes. long input_size,
  * \@Cast("int64_t") long hidden_size T_TensorTensor_TOptional, T_TensorT_TensorTensor_T_T,
  * @group nn_conv
  */
final class LSTMCell[ParamType <: FloatNN | ComplexNN: Default](
    val input_size: Int,
    val hidden_size: Int,
    val bias: Boolean = true
) extends HasParams[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options = new LSTMCellOptions(input_size.toLong, hidden_size.toLong)
  options.hidden_size().put(hidden_size.toLong)
  options.bias().put(bias)

  override private[torch] val nativeModule: LSTMCellImpl = LSTMCellImpl(options)
  nativeModule.to(paramType.toScalarType, false)

  def weight_ih(): Tensor[ParamType] = fromNative(nativeModule.weight_ih())

  def weight_hh(): Tensor[ParamType] = fromNative(nativeModule.weight_hh())

  def bias_ih(): Tensor[ParamType] = fromNative(nativeModule.bias_ih())

  def bias_hh(): Tensor[ParamType] = fromNative(nativeModule.bias_hh())

  def weight_ih(weight: Tensor[ParamType]) = nativeModule.weight_ih(weight.native)

  def weight_hh(weight: Tensor[ParamType]) = nativeModule.weight_hh(weight.native)

  def bias_ih(bias: Tensor[ParamType]) = nativeModule.bias_ih(bias.native)

  def bias_hh(bias: Tensor[ParamType]) = nativeModule.bias_hh(bias.native)

  def reset(): Unit = nativeModule.reset()

  def reset_parameters(): Unit = nativeModule.reset_parameters()

  def apply(
      t: Tensor[ParamType],
      h0: Tensor[ParamType],
      c0: Tensor[ParamType]
  ): Tuple2[Tensor[ParamType], Tensor[ParamType]] = {
    val hx = new T_TensorTensor_T(h0.native, c0.native)
//    val hx_opt = new T_TensorTensor_TOptional(hx)
    val fore = nativeModule.forward(t.native, hx)
    (fromNative(fore.get0()), fromNative(fore.get1()))
  }

  def apply(
      input: Tensor[ParamType],
      hidden_state: Tuple2[Tensor[ParamType], Tensor[ParamType]]
  ): Tuple2[Tensor[ParamType], Tensor[ParamType]] = {
    val hx = new T_TensorTensor_T(hidden_state._1.native, hidden_state._2.native)
//    val hx_opt = new T_TensorTensor_TOptional(hx)
    val fore = nativeModule.forward(input.native, hx)
    (fromNative(fore.get0()), fromNative(fore.get1()))
  }

  def apply(
      input: Tensor[ParamType],
      hidden_state: Option[Tuple2[Tensor[ParamType], Tensor[ParamType]]] = None
  ): Tuple2[Tensor[ParamType], Tensor[ParamType]] = {
    if (hidden_state.isDefined) {
      val hx = new T_TensorTensor_T(hidden_state.get._1.native, hidden_state.get._2.native)
//      val hx_opt = new T_TensorTensor_TOptional(hx)
      val fore = nativeModule.forward(input.native, hx)
      return (fromNative(fore.get0()), fromNative(fore.get1()))
    } else {
      val fore = nativeModule.forward(input.native)
      return (fromNative(fore.get0()), fromNative(fore.get1()))
    }
  }

  def apply(
      t: Tensor[ParamType],
      h0: Option[Tensor[ParamType]],
      c0: Option[Tensor[ParamType]]
  ): Tuple2[Tensor[ParamType], Tensor[ParamType]] = {
    if (h0.isDefined && c0.isDefined) {
      val hx = new T_TensorTensor_T(h0.get.native, c0.get.native)
//      val hx_opt = new T_TensorTensor_TOptional(hx)
      val fore = nativeModule.forward(t.native, hx)
      return (fromNative(fore.get0()), fromNative(fore.get1()))
    } else {

      val fore = nativeModule.forward(t.native)
      return (fromNative(fore.get0()), fromNative(fore.get1()))
    }

  }

  def weight: TensorVector =
    TensorVector(nativeModule.weight_hh(), nativeModule.weight_ih()) /// all_weights()

  def forward(
      t: Tensor[ParamType],
      h0: Tensor[ParamType],
      c0: Tensor[ParamType]
  ): Tuple2[Tensor[ParamType], Tensor[ParamType]] = {
    val hx = new T_TensorTensor_T(h0.native, c0.native)
    //    val hx_opt = new T_TensorTensor_TOptional(hx)
    val fore = nativeModule.forward(t.native, hx)
    (fromNative(fore.get0()), fromNative(fore.get1()))
  }

  def forward(
      input: Tensor[ParamType],
      hidden_state: Tuple2[Tensor[ParamType], Tensor[ParamType]]
  ): Tuple2[Tensor[ParamType], Tensor[ParamType]] = {
    val hx = new T_TensorTensor_T(hidden_state._1.native, hidden_state._2.native)
    //    val hx_opt = new T_TensorTensor_TOptional(hx)
    val fore = nativeModule.forward(input.native, hx)
    (fromNative(fore.get0()), fromNative(fore.get1()))
  }

  def forward(
      input: Tensor[ParamType],
      hidden_state: Option[Tuple2[Tensor[ParamType], Tensor[ParamType]]] = None
  ): Tuple2[Tensor[ParamType], Tensor[ParamType]] = {
    if (hidden_state.isDefined) {
      val hx = new T_TensorTensor_T(hidden_state.get._1.native, hidden_state.get._2.native)
      //      val hx_opt = new T_TensorTensor_TOptional(hx)
      val fore = nativeModule.forward(input.native, hx)
      return (fromNative(fore.get0()), fromNative(fore.get1()))
    } else {
      val fore = nativeModule.forward(input.native)
      return (fromNative(fore.get0()), fromNative(fore.get1()))
    }
  }

  def forward(
      t: Tensor[ParamType],
      h0: Option[Tensor[ParamType]],
      c0: Option[Tensor[ParamType]]
  ): Tuple2[Tensor[ParamType], Tensor[ParamType]] = {
    if (h0.isDefined && c0.isDefined) {
      val hx = new T_TensorTensor_T(h0.get.native, c0.get.native)
      //      val hx_opt = new T_TensorTensor_TOptional(hx)
      val fore = nativeModule.forward(t.native, hx)
      return (fromNative(fore.get0()), fromNative(fore.get1()))
    } else {

      val fore = nativeModule.forward(t.native)
      return (fromNative(fore.get0()), fromNative(fore.get1()))
    }

  }
  override def hasBias(): Boolean = options.bias().get()

  override def toString =
    s"${getClass.getSimpleName}(inputSize=$input_size, hiddenSize=$hidden_size bias=$bias)"

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???

object LSTMCell:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      input_size: Int,
      hidden_size: Int,
      bias: Boolean = true
  ): LSTMCell[ParamType] = new LSTMCell(input_size, hidden_size, bias)

//      val hx = new T_TensorTensor_T(h0.get.native, c0.get.native)
//      val hx_opt = new T_TensorTensor_TOptional(hx)

//    //      val hx = new T_TensorTensor_T(h0.get.native, c0.get.native)
//      //      val hx_opt = new T_TensorTensor_TOptional(hx)

//  def apply(t: Tensor[ParamType],h0: Tensor[ParamType],c0: Tensor[ParamType]): (Tensor[ParamType],Tensor[ParamType],Tensor[ParamType]) = {
//    val fore = nativeModule.forward(t.native,h0.native,c0.native)
//    val fore2 = fore.get1()
//    (fromNative(fore.get0()),fromNative(fore2.get0()), fromNative(fore2.get1()))
//  }
