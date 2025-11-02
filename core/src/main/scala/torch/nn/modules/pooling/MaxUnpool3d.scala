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
import org.bytedeco.pytorch.{MaxUnpool3dImpl, MaxUnpool3dOptions, LongVectorOptional, LongVector}
import torch.internal.NativeConverters.{fromNative, toNative}

/** Applies a 2D max pooling over an input signal composed of several input planes.
  * torch.nn.MaxUnpool3d(kernel_size, stride=None, padding=0)
  */
final class MaxUnpool3d[D <: FloatNN | ComplexNN: Default](
    val kernel_size: Int | (Int, Int) | (Int, Int, Int),
    val stride: Int | (Int, Int) | (Int, Int, Int) | Option[Int] = None,
    val padding: Int | (Int, Int) | (Int, Int, Int) = 0
) extends TensorModule[D]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options: MaxUnpool3dOptions = kernel_size match {
    case k: Int             => MaxUnpool3dOptions(toNative((k, k, k)))
    case k: (Int, Int)      => MaxUnpool3dOptions(toNative(k))
    case k: (Int, Int, Int) => MaxUnpool3dOptions(toNative(k))
  }

  stride match {
    case s: Int             => options.stride().put(toNative((s, s, s)))
    case s: (Int, Int)      => options.stride().put(toNative(s))
    case s: (Int, Int, Int) => options.stride().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
    case None               => {}
  }
  padding match {
    case s: Int             => options.padding().put(toNative((s, s, s)))
    case s: (Int, Int)      => options.padding().put(toNative(s))
    case s: (Int, Int, Int) => options.padding().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
  }
  kernel_size match {
    case s: Int        => options.kernel_size().put(toNative((s, s, s)))
    case s: (Int, Int) => options.kernel_size().put(toNative(s))
    case s: (Int, Int, Int) =>
      options.kernel_size().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
  }

  override private[torch] val nativeModule: MaxUnpool3dImpl = MaxUnpool3dImpl(options)

  def reset(): Unit = nativeModule.reset()

  override def hasBias(): Boolean = false

  override def toString(): String =
    s"${getClass.getSimpleName}(kernelSize=$kernel_size, stride=$stride, padding=$padding)"

  def apply(input: Tensor[D], indices: Tensor[Int64]): Tensor[D] = fromNative(
    nativeModule.forward(input.native, indices.native)
  )
  def apply(input: Tensor[D], indices: Tensor[Int64], outputSize: Array[Int]): Tensor[D] =
    val out = outputSize.map(_.toLong)
    fromNative(
      nativeModule.forward(input.native, indices.native, LongVectorOptional(LongVector(out*)))
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

object MaxUnpool3d:
  def apply[D <: FloatNN | ComplexNN: Default](
      kernel_size: Int | (Int, Int) | (Int, Int, Int),
      stride: Int | (Int, Int) | (Int, Int, Int) | Option[Int] = None,
      padding: Int | (Int, Int) | (Int, Int, Int) = 0
  ): MaxUnpool3d[D] =
    new MaxUnpool3d(kernel_size, stride, padding)

//  def  apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))
