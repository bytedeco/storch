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
package pad

import org.bytedeco.javacpp.{LongPointer}

import org.bytedeco.pytorch.{ReflectionPad3dImpl, ReflectionPad3dOptions}
import torch.internal.NativeConverters.{fromNative, toNative}

/** Applies a 2D adaptive average pooling over an input signal composed of several input planes.
  *
  * The output is of size H x W, for any input size. The number of output features is equal to the
  * number of input planes.
  */
final class ReflectionPad3d[ParamType <: FloatNN | ComplexNN: Default](
    val padding: Int | (Int, Int, Int, Int, Int, Int)
) extends HasParams[ParamType]
    with TensorModule[ParamType] {
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  val paddingNative = padding match {
    case (top, bottom, left, right, int, float) => toNative(top, bottom, left, right, int, float)
    case x: Int => LongPointer(Array(x.toLong, x.toLong, x.toLong, x.toLong, x.toLong, x.toLong)*)
    case _ => throw new IllegalArgumentException("padding must be a tuple of 2, 4 or 8 integers")
  }

  private val options: ReflectionPad3dOptions = ReflectionPad3dOptions(paddingNative)
  options.padding().put(paddingNative)

  override protected[torch] val nativeModule: ReflectionPad3dImpl = ReflectionPad3dImpl(
    options
  )

  def reset(): Unit = nativeModule.reset()
  override def hasBias(): Boolean = false

  override def toString =
    s"${getClass.getSimpleName}(padding = ${padding}"

  def apply(t: Tensor[ParamType]): Tensor[ParamType] = fromNative(
    nativeModule.forward(t.native)
  )
  def forward(input: Tensor[ParamType]): Tensor[ParamType] = fromNative(
    nativeModule.forward(input.native)
  )
}

object ReflectionPad3d:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      padding: Int | (Int, Int, Int, Int, Int, Int)
  ): ReflectionPad3d[ParamType] =
    new ReflectionPad3d(padding)
