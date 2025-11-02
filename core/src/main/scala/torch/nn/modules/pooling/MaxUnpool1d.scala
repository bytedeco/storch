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
import org.bytedeco.pytorch.{MaxUnpool1dImpl, MaxUnpool1dOptions, LongVectorOptional, LongVector}
import torch.internal.NativeConverters.{fromNative, toNative}

/** Applies a 2D max pooling over an input signal composed of several input planes.
  * torch.nn.MaxUnpool1d(kernel_size, stride=None, padding=0)
  */
final class MaxUnpool1d[D <: FloatNN | ComplexNN: Default](
    val kernel_size: Int | (Int, Int),
    val stride: Int | (Int, Int) | Option[Int] = None,
    val padding: Int | (Int, Int) = 0
) extends TensorModule[D]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options: MaxUnpool1dOptions = MaxUnpool1dOptions(toNative(kernel_size))
  stride match {
    case s: Int        => options.stride().put(Array(s.toLong, s.toLong)*)
    case s: (Int, Int) => options.stride().put(Array(s._1.toLong, s._2.toLong)*)
    case None          => {}
  }
  padding match {
    case s: Int        => options.padding().put(Array(s.toLong, s.toLong)*)
    case s: (Int, Int) => options.padding().put(Array(s._1.toLong, s._2.toLong)*)
  }
  kernel_size match {
    case s: Int        => options.kernel_size().put(Array(s.toLong, s.toLong)*)
    case s: (Int, Int) => options.kernel_size().put(Array(s._1.toLong, s._2.toLong)*)
  }

  override private[torch] val nativeModule: MaxUnpool1dImpl = MaxUnpool1dImpl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  override def toString(): String =
    s"${getClass.getSimpleName}(kernelSize=$kernel_size, stride=$stride, padding=$padding)"

  def apply(input: Tensor[D], indices: Tensor[Int64], outputSize: Array[Int]): Tensor[D] =
    val out = outputSize.map(_.toLong)
    fromNative(
      nativeModule.forward(input.native, indices.native, LongVectorOptional(LongVector(out*)))
    )

  def apply(input: Tensor[D], indices: Tensor[Int64]): Tensor[D] = fromNative(
    nativeModule.forward(input.native, indices.native)
  )

  def forward(input: Tensor[D], indices: Tensor[Int64]): Tensor[D] = fromNative(
    nativeModule.forward(input.native, indices.native)
  )

  def forward(input: Tensor[D], indices: Tensor[Int64], outputSize: Array[Int]): Tensor[D] =
    val out = outputSize.map(_.toLong)
    fromNative(
      nativeModule.forward(input.native, indices.native, LongVectorOptional(LongVector(out*)))
    )

  override def apply(v1: Tensor[D]): Tensor[D] = ???

object MaxUnpool1d:
  def apply[D <: FloatNN | ComplexNN: Default](
      kernel_size: Int | (Int, Int),
      stride: Int | (Int, Int) | Option[Int] = None,
      padding: Int | (Int, Int) = 0
  ): MaxUnpool1d[D] = new MaxUnpool1d(kernel_size, stride, padding)

//  options.padding().put(toNative(padding))
//  options.kernel_size().put(toNative(kernelSize))
//  options.stride().put(toNative(stride))
