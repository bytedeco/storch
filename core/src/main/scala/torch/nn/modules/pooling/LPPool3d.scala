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

import org.bytedeco.javacpp.{DoublePointer}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{LPPool3dImpl, LPPool3dOptions}
import torch.internal.NativeConverters.{fromNative, toNative}

/** Applies a 2D max pooling over an input signal composed of several input planes. */
final class LPPool3d[D <: FloatNN | ComplexNN: Default](
    val norm_type: Float,
    val kernel_size: Int | (Int, Int, Int),
    val stride: Int | (Int, Int, Int) | Option[Int] = None,
    val ceil_mode: Boolean = false
) extends TensorModule[D]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options: LPPool3dOptions = LPPool3dOptions(toNative(kernel_size))

  stride match {
    case s: Int             => options.stride().put(Array(s.toLong, s.toLong, s.toLong)*)
    case s: (Int, Int, Int) => options.stride().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
    case None               => {}
  }
  kernel_size match {
    case s: Int => options.kernel_size().put(Array(s.toLong, s.toLong, s.toLong)*)
    case s: (Int, Int, Int) =>
      options.kernel_size().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
  }

  options.ceil_mode().put(ceil_mode)
  options.norm_type().put(DoublePointer(1).put(norm_type.toDouble))

  override private[torch] val nativeModule: LPPool3dImpl = LPPool3dImpl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  override def toString(): String =
    s"${getClass.getSimpleName}(kernelSize=$kernel_size, stride=$stride, normType=$norm_type,  ceilMode=$ceil_mode)"

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))
  def forward(input: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(input.native))

object LPPool3d:
  def apply[D <: FloatNN | ComplexNN: Default](
      norm_type: Float,
      kernel_size: Int | (Int, Int, Int),
      stride: Int | (Int, Int, Int) | Option[Int] = None,
      ceil_mode: Boolean = false
  ): LPPool3d[D] =
    new LPPool3d[D](norm_type, kernel_size, stride, ceil_mode)

//  private val options: LPPool3dOptions = kernelSize match {
//    case k: Int             => LPPool3dOptions(toNative((k, k, k)))
//    case k: (Int, Int, Int) => LPPool3dOptions(toNative(k))
//  }
