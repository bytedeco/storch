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
package pooling

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{MaxPool3dImpl, MaxPool3dOptions, T_TensorTensor_T}
import torch.internal.NativeConverters.{fromNative, toNative}

/** Applies a 2D max pooling over an input signal composed of several input planes. */
final class MaxPool3d[D <: FloatNN | ComplexNN: Default](
    val kernel_size: Int | (Int, Int, Int),
    val stride: Int | (Int, Int, Int) | Option[Int] = None,
    val padding: Int | (Int, Int, Int) = 0,
    val dilation: Int | (Int, Int, Int) = 1,
    val return_indices: Boolean = false,
    val ceil_mode: Boolean = false
) extends TensorModule[D]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options: MaxPool3dOptions = kernel_size match {
    case k: Int             => MaxPool3dOptions(toNative((k, k, k)))
    case k: (Int, Int, Int) => MaxPool3dOptions(toNative(k))
  }
  stride match {
    case s: Int             => options.stride().put(toNative((s, s, s)))
    case s: (Int, Int, Int) => options.stride().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
    case None               => {}
  }
  kernel_size match {
    case s: Int => options.kernel_size().put(toNative((s, s, s)))
    case s: (Int, Int, Int) =>
      options.kernel_size().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
  }
  padding match {
    case s: Int             => options.padding().put(toNative((s, s, s)))
    case s: (Int, Int, Int) => options.padding().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
  }
  dilation match {
    case s: Int             => options.dilation().put(toNative((s, s, s)))
    case s: (Int, Int, Int) => options.dilation().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
  }

  options.ceil_mode().put(ceil_mode)
  override private[torch] val nativeModule: MaxPool3dImpl = MaxPool3dImpl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  override def toString(): String =
    s"${getClass.getSimpleName}(kernelSize=$kernel_size, stride=$stride, padding=$padding, dilation=$dilation, ceilMode=$ceil_mode)"

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))
  def forward(input: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(input.native))

  def forward_with_indices(t: Tensor[D]): (Tensor[D], Tensor[Int64]) =
    val outputWithIndices = nativeModule.forward_with_indices(t.native)
    (fromNative(outputWithIndices.get0()), fromNative(outputWithIndices.get1()))

object MaxPool3d:
  def apply[D <: FloatNN | ComplexNN: Default](
      kernel_size: Int | (Int, Int, Int),
      stride: Int | (Int, Int, Int) | Option[Int] = None,
      padding: Int | (Int, Int, Int) = 0,
      dilation: Int | (Int, Int, Int) = 1,
      return_indices: Boolean = false,
      ceil_mode: Boolean = false
  ): MaxPool3d[D] =
    new MaxPool3d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
