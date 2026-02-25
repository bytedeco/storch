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

import org.bytedeco.pytorch.{ConstantPad1dImpl, ConstantPad1dOptions}
import torch.internal.NativeConverters.{fromNative, toNative}

/** Applies a 2D adaptive average pooling over an input signal composed of several input planes.
  * FloatNN | ComplexNN: Default The output is of size H x W, for any input size. The number of
  * output features is equal to the number of input planes. FloatNN | ComplexNN: Default
  */
final class ConstantPad1d[ParamType <: FloatNN | ComplexNN: Default](
    val padding: Int | (Int, Int),
    val value: Float | Double
) extends HasParams[ParamType]
    with TensorModule[ParamType] {
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options: ConstantPad1dOptions = ConstantPad1dOptions(toNative(padding))
  options.padding().put(toNative(padding))

  value match {
    case v: Float  => options.value().put(v.toDouble)
    case v: Double => options.value().put(v)
  }
  override protected[torch] val nativeModule: ConstantPad1dImpl = ConstantPad1dImpl(
    options
  )

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()
  def apply(t: Tensor[ParamType]): Tensor[ParamType] = fromNative(
    nativeModule.forward(t.native)
  )
  def forward(input: Tensor[ParamType]): Tensor[ParamType] = fromNative(
    nativeModule.forward(input.native)
  )

  override def toString =
    s"${getClass.getSimpleName}(padding = ${padding}, value = ${value})"
}

object ConstantPad1d:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      padding: Int | (Int, Int),
      value: Float | Double
  ): ConstantPad1d[ParamType] =
    new ConstantPad1d[ParamType](padding, value)
