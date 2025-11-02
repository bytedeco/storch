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
import org.bytedeco.pytorch.{PackedSequence}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{RNNImpl, RNNOptions, kTanh, kReLU}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn.modules.recurrent.RNN.RNNNonlinearity

/** Applies a 2D convolution over an input signal composed of several input planes. long input_size,
  * \@Cast("int64_t") long hidden_size // RNNNonlinearity = RNNNonlinearity.kTanh,
  * @group nn_conv
  */
final class RNN[ParamType <: FloatNN | ComplexNN: Default](
    val input_size: Int,
    val hidden_size: Int,
    val num_layers: Int = 1,
    val nonlinearity: String | RNNNonlinearity = "tanh",
    val bias: Boolean = true,
    val batch_first: Boolean = false,
    val dropout: Float | Double = 0f,
    val bidirectional: Boolean = false
) extends HasParams[ParamType]
    with TensorModule[ParamType]:
  type PackedSequenceTensor = (PackedSequence, Tensor[ParamType])
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options = new RNNOptions(input_size.toLong, hidden_size.toLong)
  options.bias().put(bias)
  dropout match {
    case p: Float  => options.dropout().put(p.toDouble)
    case p: Double => options.dropout().put(p)
  }
  options.batch_first().put(batch_first)
  options.bidirectional().put(bidirectional)
  options.input_size().put(LongPointer(1).put(input_size.toLong))
  options.hidden_size().put(LongPointer(1).put(hidden_size.toLong))
  options.num_layers().put(LongPointer(1).put(num_layers.toLong))

  nonlinearity match
    case "tanh" | "Tanh"       => options.nonlinearity().put(new kTanh)
    case "relu" | "Relu"       => options.nonlinearity().put(new kReLU)
    case RNNNonlinearity.kTanh => options.nonlinearity().put(new kTanh)
    case RNNNonlinearity.kReLU => options.nonlinearity().put(new kReLU)

  override private[torch] val nativeModule: RNNImpl = RNNImpl(options)
  nativeModule.to(paramType.toScalarType, false)

  def apply(
      input: Tensor[ParamType],
      hx: Tensor[ParamType]
  ): Tuple2[Tensor[ParamType], Tensor[ParamType]] = {
    val fore = nativeModule.forward(input.native, hx.native)
    Tuple2(fromNative(fore.get0()), fromNative(fore.get1()))
  }
  def forward(
      input: Tensor[ParamType],
      hx: Tensor[ParamType]
  ): Tuple2[Tensor[ParamType], Tensor[ParamType]] = {
    val fore = nativeModule.forward(input.native, hx.native)
    Tuple2(fromNative(fore.get0()), fromNative(fore.get1()))
  }

  def apply(
      input: Tensor[ParamType],
      hx: Option[Tensor[ParamType]] = None
  ): Tuple2[Tensor[ParamType], Tensor[ParamType]] = {
    val fore =
      if hx.isDefined then nativeModule.forward(input.native, hx.get.native)
      else nativeModule.forward(input.native)
    Tuple2(fromNative(fore.get0()), fromNative(fore.get1()))
  }
  def forward(
      input: Tensor[ParamType],
      hx: Option[Tensor[ParamType]] = None
  ): Tuple2[Tensor[ParamType], Tensor[ParamType]] = {
    val fore =
      if hx.isDefined then nativeModule.forward(input.native, hx.get.native)
      else nativeModule.forward(input.native)
    Tuple2(fromNative(fore.get0()), fromNative(fore.get1()))
  }

  def apply(packed_input: PackedSequence): PackedSequenceTensor = {

    val output = nativeModule.forward_with_packed_input(packed_input)
    (output.get0(), fromNative(output.get1()))
  }
  def forward(packed_input: PackedSequence): PackedSequenceTensor = {
    val output = nativeModule.forward_with_packed_input(packed_input)
    (output.get0(), fromNative(output.get1()))
  }

  def apply(packed_input: PackedSequence, hx: Tensor[ParamType]): PackedSequenceTensor = {

    val output = nativeModule.forward_with_packed_input(packed_input, hx.native)
    (output.get0(), fromNative(output.get1()))
  }

  def forward(packed_input: PackedSequence, hx: Tensor[ParamType]): PackedSequenceTensor = {

    val output = nativeModule.forward_with_packed_input(packed_input, hx.native)
    (output.get0(), fromNative(output.get1()))
  }

  def forward_with_packed_input(packed_input: PackedSequence): PackedSequenceTensor = {

    val output = nativeModule.forward_with_packed_input(packed_input)
    (output.get0(), fromNative(output.get1()))
  }

  def forward_with_packed_input(
      packed_input: PackedSequence,
      hx: Tensor[ParamType]
  ): PackedSequenceTensor = {

    val output = nativeModule.forward_with_packed_input(packed_input, hx.native)
    (output.get0(), fromNative(output.get1()))
  }

  def reset(): Unit = nativeModule.reset()

  def reset_parameters(): Unit = nativeModule.reset_parameters()

  def all_weights(): Seq[Tensor[ParamType]] = {
    val vec = nativeModule.all_weights()
    torch.tensorVectorToSeqTensor(vec)
  }

  def weights = all_weights() // : TensorVector = nativeModule.all_weights()

  override def hasBias(): Boolean = options.bias().get()

  override def toString =
    s"${getClass.getSimpleName}(inputSize=$input_size, hiddenSize=$hidden_size,numLayers=${num_layers}, nonLinearity ${nonlinearity},batchFirst = ${batch_first},dropout = ${dropout},bidirectional = ${bidirectional} bias=$bias)"

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???

object RNN:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      input_size: Int,
      hidden_size: Int,
      num_layers: Int = 1,
      nonlinearity: String | RNNNonlinearity = "tanh",
      bias: Boolean = true,
      batch_first: Boolean = false,
      dropout: Float | Double = 0f,
      bidirectional: Boolean = false
  ): RNN[ParamType] = new RNN(
    input_size,
    hidden_size,
    num_layers,
    nonlinearity,
    bias,
    batch_first,
    dropout,
    bidirectional
  )
  enum RNNNonlinearity:
    case kTanh, kReLU
