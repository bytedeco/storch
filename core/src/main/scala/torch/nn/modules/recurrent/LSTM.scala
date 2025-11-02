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

import org.bytedeco.javacpp.{LongPointer}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{LSTMImpl, PackedSequence, LSTMOptions, T_TensorTensor_T}
import torch.internal.NativeConverters.{fromNative, toNative}

/** Applies a 2D convolution over an input signal composed of several input planes. long input_size,
  * \@Cast("int64_t") long hidden_size T_TensorTensor_TOptional, T_TensorT_TensorTensor_T_T,
  * TensorVector,
  *
  * @group nn_conv
  */
object LSTM:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      input_size: Int,
      hidden_size: Int,
      num_layers: Int = 1,
      bias: Boolean = true,
      batch_first: Boolean = false,
      dropout: Float | Double = 0.1f,
      bidirectional: Boolean = false,
      proj_size: Int = 0
  ): LSTM[ParamType] =
    new LSTM(
      input_size,
      hidden_size,
      num_layers,
      bias,
      batch_first,
      dropout,
      bidirectional,
      proj_size
    )

final class LSTM[ParamType <: FloatNN | ComplexNN: Default](
    val input_size: Int,
    val hidden_size: Int,
    val num_layers: Int = 1,
    val bias: Boolean = true,
    val batch_first: Boolean = false,
    val dropout: Float | Double = 0.1f,
    val bidirectional: Boolean = false,
    val proj_size: Int = 0
) extends HasParams[ParamType]
    with TensorModule[ParamType]:
  type PackedSequenceTensorTensor =
    (PackedSequence, Tensor[ParamType], Tensor[ParamType]) // T_PackedSequenceT_TensorTensor_T_T
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options = new LSTMOptions(input_size.toLong, hidden_size.toLong)
  options.input_size().put(LongPointer(1).put(input_size.toLong))
  options.hidden_size().put(LongPointer(1).put(hidden_size.toLong))
  options.num_layers().put(LongPointer(1).put(num_layers.toLong))
  dropout match {
    case p: Float  => options.dropout().put(p.toDouble)
    case p: Double => options.dropout().put(p)
  }
  options.bias().put(bias)
  options.batch_first().put(batch_first)
  options.bidirectional().put(bidirectional)

  override def toString =
    s"${getClass.getSimpleName}(inputSize=$input_size, hiddenSize=$hidden_size,numLayers=${num_layers},batchFirst = ${batch_first},dropout = ${dropout},bidirectional = ${bidirectional} bias=$bias)"

  override private[torch] val nativeModule: LSTMImpl = LSTMImpl(options)
  nativeModule.to(paramType.toScalarType, false)

  def apply(
      t: Tensor[ParamType],
      h0: Tensor[ParamType],
      c0: Tensor[ParamType]
  ): Tuple3[Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]] = {
    val hx = new T_TensorTensor_T(h0.native, c0.native)
//    val hx_opt = new T_TensorTensor_TOptional(hx)
    val fore = nativeModule.forward(t.native, hx)
    val fore2 = fore.get1()
    (fromNative(fore.get0()), fromNative(fore2.get0()), fromNative(fore2.get1()))
  }

  def apply(
      input: Tensor[ParamType],
      hidden_state: Tuple2[Tensor[ParamType], Tensor[ParamType]]
  ): Tuple2[Tensor[ParamType], Tuple2[Tensor[ParamType], Tensor[ParamType]]] = {
    val hx = new T_TensorTensor_T(hidden_state._1.native, hidden_state._2.native)
//    val hx_opt = new T_TensorTensor_TOptional(hx)
    val fore = nativeModule.forward(input.native, hx)
    val fore2 = fore.get1()
    (fromNative(fore.get0()), (fromNative(fore2.get0()), fromNative(fore2.get1())))
  }

  def apply(
      t: Tensor[ParamType],
      h0: Option[Tensor[ParamType]] = None,
      c0: Option[Tensor[ParamType]] = None
  ): Tuple3[Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]] = {
    val hx = new T_TensorTensor_T(h0.get.native, c0.get.native)
//    val hx_opt = new T_TensorTensor_TOptional(hx)
    val fore = nativeModule.forward(t.native, hx)
    val fore2 = fore.get1()
    (fromNative(fore.get0()), fromNative(fore2.get0()), fromNative(fore2.get1()))
  }

  def apply(
      input: Tensor[ParamType],
      hidden_state: Option[Tuple2[Tensor[ParamType], Tensor[ParamType]]]
  ): Tuple2[Tensor[ParamType], Tuple2[Tensor[ParamType], Tensor[ParamType]]] = {
    val hx = new T_TensorTensor_T(hidden_state.get._1.native, hidden_state.get._2.native)
//    val hx_opt = new T_TensorTensor_TOptional(hx)
    val fore = nativeModule.forward(input.native, hx)
    val fore2 = fore.get1()
    (fromNative(fore.get0()), (fromNative(fore2.get0()), fromNative(fore2.get1())))
  }

  def apply(packed_input: PackedSequence): PackedSequenceTensorTensor = {

    val output = nativeModule.forward_with_packed_input(packed_input)
    (output.get0(), fromNative(output.get1().get0()), fromNative(output.get1().get1()))

  }

  def apply(
      packed_input: PackedSequence,
      hx: Tensor[ParamType],
      cx: Tensor[ParamType]
  ): PackedSequenceTensorTensor = {
    val hxx = new T_TensorTensor_T(hx.native, cx.native)
//    val hxx_opt = new T_TensorTensor_TOptional(hxx)
    val output = nativeModule.forward_with_packed_input(packed_input, hxx)
    (output.get0(), fromNative(output.get1().get0()), fromNative(output.get1().get1()))
  }

  def forward_with_packed_input(packed_input: PackedSequence): PackedSequenceTensorTensor = {

    val output = nativeModule.forward_with_packed_input(packed_input)
    (output.get0(), fromNative(output.get1().get0()), fromNative(output.get1().get1()))

  }

  def forward_with_packed_input(
      packed_input: PackedSequence,
      hx: Tensor[ParamType],
      cx: Tensor[ParamType]
  ): PackedSequenceTensorTensor = {
    val hxx = new T_TensorTensor_T(hx.native, cx.native)
//    val hxx_opt = new T_TensorTensor_TOptional(hxx)
    val output = nativeModule.forward_with_packed_input(packed_input, hxx)
    (output.get0(), fromNative(output.get1().get0()), fromNative(output.get1().get1()))
  }

  def forward(
      t: Tensor[ParamType],
      h0: Tensor[ParamType],
      c0: Tensor[ParamType]
  ): Tuple3[Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]] = {
    val hx = new T_TensorTensor_T(h0.native, c0.native)
    //    val hx_opt = new T_TensorTensor_TOptional(hx)
    val fore = nativeModule.forward(t.native, hx)
    val fore2 = fore.get1()
    (fromNative(fore.get0()), fromNative(fore2.get0()), fromNative(fore2.get1()))
  }

  def forward(
      input: Tensor[ParamType],
      hidden_state: Tuple2[Tensor[ParamType], Tensor[ParamType]]
  ): Tuple2[Tensor[ParamType], Tuple2[Tensor[ParamType], Tensor[ParamType]]] = {
    val hx = new T_TensorTensor_T(hidden_state._1.native, hidden_state._2.native)
    //    val hx_opt = new T_TensorTensor_TOptional(hx)
    val fore = nativeModule.forward(input.native, hx)
    val fore2 = fore.get1()
    (fromNative(fore.get0()), (fromNative(fore2.get0()), fromNative(fore2.get1())))
  }

  def forward(
      t: Tensor[ParamType],
      h0: Option[Tensor[ParamType]] = None,
      c0: Option[Tensor[ParamType]] = None
  ): Tuple3[Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]] = {
    val hx = new T_TensorTensor_T(h0.get.native, c0.get.native)
    //    val hx_opt = new T_TensorTensor_TOptional(hx)
    val fore = nativeModule.forward(t.native, hx)
    val fore2 = fore.get1()
    (fromNative(fore.get0()), fromNative(fore2.get0()), fromNative(fore2.get1()))
  }

  def forward(
      input: Tensor[ParamType],
      hidden_state: Option[Tuple2[Tensor[ParamType], Tensor[ParamType]]]
  ): Tuple2[Tensor[ParamType], Tuple2[Tensor[ParamType], Tensor[ParamType]]] = {
    val hx = new T_TensorTensor_T(hidden_state.get._1.native, hidden_state.get._2.native)
    //    val hx_opt = new T_TensorTensor_TOptional(hx)
    val fore = nativeModule.forward(input.native, hx)
    val fore2 = fore.get1()
    (fromNative(fore.get0()), (fromNative(fore2.get0()), fromNative(fore2.get1())))
  }

  def forward(packed_input: PackedSequence): PackedSequenceTensorTensor = {

    val output = nativeModule.forward_with_packed_input(packed_input)
    (output.get0(), fromNative(output.get1().get0()), fromNative(output.get1().get1()))

  }

  def forward(
      packed_input: PackedSequence,
      hx: Tensor[ParamType],
      cx: Tensor[ParamType]
  ): PackedSequenceTensorTensor = {
    val hxx = new T_TensorTensor_T(hx.native, cx.native)
    //    val hxx_opt = new T_TensorTensor_TOptional(hxx)
    val output = nativeModule.forward_with_packed_input(packed_input, hxx)
    (output.get0(), fromNative(output.get1().get0()), fromNative(output.get1().get1()))
  }

  def all_weights(): Seq[Tensor[ParamType]] = {
    val vec = nativeModule.all_weights()
    torch.tensorVectorToSeqTensor(vec)
  }
  def weights = all_weights()

  def reset(): Unit = nativeModule.reset()

  def reset_parameters(): Unit = nativeModule.reset_parameters()

  override def hasBias(): Boolean = options.bias().get()

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???

//options.hidden_size().put(hiddenSize.toLong)
//options.num_layers().put(numLayers)

//  def apply(t: Tensor[ParamType],h0: Tensor[ParamType],c0: Tensor[ParamType]): (Tensor[ParamType],Tensor[ParamType],Tensor[ParamType]) = {
//    val fore = nativeModule.forward(t.native,h0.native,c0.native)
//    val fore2 = fore.get1()
//    (fromNative(fore.get0()),fromNative(fore2.get0()), fromNative(fore2.get1()))
//  }
