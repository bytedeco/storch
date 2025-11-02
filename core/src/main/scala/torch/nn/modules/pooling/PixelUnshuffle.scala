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

import org.bytedeco.javacpp.{LongPointer}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{PixelUnshuffleImpl, PixelUnshuffleOptions}
import torch.internal.NativeConverters.{fromNative, toNative}

/** Applies a 2D max pooling over an input signal composed of several input planes. */
final class PixelUnshuffle[D <: FloatNN | ComplexNN: Default](
    val downscale_factor: Int
) extends TensorModule[D]:

  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options: PixelUnshuffleOptions = PixelUnshuffleOptions(
    LongPointer(downscale_factor.toLong)
  )
  options.downscale_factor.put(LongPointer(downscale_factor))

  override private[torch] val nativeModule: PixelUnshuffleImpl = PixelUnshuffleImpl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  override def toString(): String =
    s"${getClass.getSimpleName}(downscaleFactor=$downscale_factor)"

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))
  def forward(input: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(input.native))

object PixelUnshuffle:
  def apply[D <: FloatNN | ComplexNN: Default](downscale_factor: Int): PixelUnshuffle[D] =
    new PixelUnshuffle[D](downscale_factor)

//  private val options: PixelUnshuffleOptions = PixelUnshuffleOptions(toNative(downscaleFactor))
