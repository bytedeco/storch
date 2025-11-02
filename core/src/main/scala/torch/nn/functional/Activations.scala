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
package functional

import Derive.derive
import org.bytedeco.pytorch
import org.bytedeco.javacpp.BytePointer
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.fromNative
import org.bytedeco.pytorch.{
  RReLUFuncOptions,
  GELUOptions,
  SoftminFuncOptions,
  GumbelSoftmaxFuncOptions,
  ScalarTypeOptional,
  Scalar
}

private[torch] trait Activations {

  /** Applies a softmax followed by a logarithm.
    *
    * While mathematically equivalent to log(softmax(x)), doing these two operations separately is
    * slower and numerically unstable. This function uses an alternative formulation to compute the
    * output and gradient correctly.
    *
    * See `torch.nn.LogSoftmax` for more details.
    *
    * @group nn_activation
    */
  def logSoftmax[In <: DType, Out <: FloatNN | Derive](
      input: Tensor[In],
      dim: Int,
      dtype: Out = derive
  ): Tensor[DTypeOrDeriveFromTensor[In, Out]] =
    val derivedDType = dtype match
      case _: Derive => input.dtype
      case d: DType  => d
    val nativeDType =
      if dtype == input.dtype then ScalarTypeOptional()
      else ScalarTypeOptional(derivedDType.toScalarType)
    fromNative(torchNative.log_softmax(input.native, dim.toLong, nativeDType))

  def log_softmax[In <: DType, Out <: FloatNN | Derive](
      input: Tensor[In],
      dim: Int,
      dtype: Out = derive
  ) = logSoftmax(input, dim, dtype)

  /** Applies the rectified linear unit function element-wise.
    *
    * See [[torch.nn.ReLU]] for more details.
    *
    * @group nn_activation
    */
  def relu[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.relu(input.native))

  /** Applies the element-wise function $\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}$
    *
    * See `torch.nn.Sigmoid` for more details.
    *
    * @group nn_activation
    */
  def sigmoid[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(
    torchNative.sigmoid(input.native)
  )

  /** Applies the Sigmoid Linear Unit (SiLU) function, element-wise. The SiLU function is also known
    * as the swish function.
    *
    * @group nn_activation
    */
//  def silu[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.silu(input.native))

  /** Applies a softmax function.
    *
    * @group nn_activation
    */
  def softmax[In <: DType, Out <: FloatNN | Derive](
      input: Tensor[In],
      dim: Int,
      dtype: Out = derive
  ): Tensor[DTypeOrDeriveFromTensor[In, Out]] =
    val derivedDType = dtype match
      case _: Derive => input.dtype
      case d: DType  => d
    val nativeDType =
      if dtype == input.dtype then ScalarTypeOptional()
      else ScalarTypeOptional(derivedDType.toScalarType)
    fromNative(torchNative.softmax(input.native, dim.toLong, nativeDType))

  def relu_[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.relu_(input.native))

  def hardtanh[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(
    torchNative.hardtanh(input.native)
  )

  def hardtanh_[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(
    torchNative.hardtanh_(input.native)
  )

  def relu6[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.relu6(input.native))

  def elu[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.elu(input.native))

  def elu_[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.elu_(input.native))

  def leaky_relu[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(
    torchNative.leaky_relu(input.native)
  )

  def leaky_relu_[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(
    torchNative.leaky_relu_(input.native)
  )

  def prelu[D <: DType](input: Tensor[D], weight: Tensor[D]): Tensor[D] = fromNative(
    torchNative.prelu(input.native, weight.native)
  )

  def rrelu[D <: DType](
      input: Tensor[D],
      lower: Double = 0.125,
      upper: Double = 0.3333333333333333,
      training: Boolean = false,
      inplace: Boolean = false
  ): Tensor[D] = {
    val options = RReLUFuncOptions()
    options.lower().put(lower)
    options.upper().put(upper)
    options.training().put(training)
    options.inplace().put(inplace)
    fromNative(torchNative.rrelu(input.native, options))
  }

  def celu[D <: DType](input: Tensor[D], alpha: Double = 1.0): Tensor[D] = fromNative(
    torchNative.celu(input.native, Scalar(alpha))
  )

  def celu_[D <: DType](input: Tensor[D], alpha: Double = 1.0): Tensor[D] = fromNative(
    torchNative.celu_(input.native, Scalar(alpha))
  )

  def selu[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.selu(input.native))

  def selu_[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.selu_(input.native))

  def glu[D <: DType](input: Tensor[D], dim: Int): Tensor[D] = fromNative(
    torchNative.glu(input.native, dim.toLong)
  )

  def gelu[D <: DType](input: Tensor[D], approximate: String = "none"): Tensor[D] = {
    require(approximate == "none" || approximate == "tanh", "approximate must be none or tanh")
    val options = GELUOptions()
    options.approximate().put(BytePointer(approximate.toLowerCase))
    fromNative(
      torchNative.gelu(input.native, options)
    )
  }

  def softplus[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(
    torchNative.softplus(input.native)
  )

  def hardshrink[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(
    torchNative.hardshrink(input.native)
  )

  def tanhshrink[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(
    torchNative.tanhshrink(input.native)
  )

  def softshrink[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(
    torchNative.softshrink(input.native)
  )

  def softsign[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(
    torchNative.softsign(input.native)
  )

  def softmin[D <: DType](input: Tensor[D], dim: Int, dtype: DType): Tensor[D] = {
    val options = SoftminFuncOptions(dim.toLong)
    options.dtype().put(ScalarTypeOptional(dtype.toScalarType))
    fromNative(
      torchNative.softmin(input.native, options)
    )
  }

  def gumbel_softmax[D <: DType](
      input: Tensor[D],
      dim: Int,
      hard: Boolean,
      tau: Double
  ): Tensor[D] = {
    val options = GumbelSoftmaxFuncOptions()
    options.dim().put(dim)
    options.hard().put(hard)
    options.tau().put(tau)
    fromNative(torchNative.gumbel_softmax(input.native, options))
  }

  def tanh[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.tanh(input.native))

  def hardsigmoid[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(
    torchNative.hardsigmoid(input.native)
  )

  def silu[D <: DType](input: Tensor[D]): Tensor[D] = {

    fromNative(torchNative.silu(input.native))
  }

  def mish[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.mish(input.native))

}

//  def relu[TT <: DType](@Const @ByRef input: Tensor[TT]): Tensor[TT] = fromNative(torchNative.relu(input.native))

//  def rrelu_[D <: DType](input: Tensor[D], lower: Double, upper: Double): Tensor[D] = {
//
//    fromNative(
//      torchNative.rrelu_(input.native, lower, upper)
//    )
//  }

//  def softsign_[D <: DType](input: Tensor[D]): Tensor[D] = {
//
//    fromNative(torchNative.softsign_(input.native))
//  }

//  def softmax[D <: DType](input: Tensor[D], dim: Long, dtype: DType = Float): Tensor[D] = fromNative(
//    torchNative.softmax(input.native, dim, dtype.toScalarType)
//  )

//  def sigmoid[D <: DType](input: Tensor[D]): Tensor[D] = {
//
//    fromNative(torchNative.sigmoid(input.native))
//  }
