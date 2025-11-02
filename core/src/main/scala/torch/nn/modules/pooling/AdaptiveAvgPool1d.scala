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

import org.bytedeco.pytorch.{AdaptiveAvgPool1dImpl, AdaptiveAvgPool1dOptions}
import org.bytedeco.pytorch

import torch.internal.NativeConverters.{fromNative, toNative, toOptional}
import org.bytedeco.pytorch.LongOptionalVector

/** Applies a 2D adaptive average pooling over an input signal composed of several input planes.
  *
  * The output is of size H x W, for any input size. The number of output features is equal to the
  * number of input planes.
  */
final class AdaptiveAvgPool1d[ParamType <: FloatNN | ComplexNN: Default](
    val output_size: Int | Option[Int] | (Option[Int], Option[Int]) | (Int, Int)
) extends HasParams[ParamType]
    with TensorModule[ParamType] {
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private def nativeOutputSize = output_size match
    case (h: Int, w: Int) =>
      toNative(h, w)
    case x: Int => toNative(x)
    case (h: Option[Int @unchecked], w: Option[Int @unchecked]) =>
      new LongOptionalVector(h.toOptional, w.toOptional)
    case x: Option[Int] =>
      new LongOptionalVector(x.toOptional, x.toOptional)

  private val options: AdaptiveAvgPool1dOptions = AdaptiveAvgPool1dOptions(nativeOutputSize)
  options.output_size().put(nativeOutputSize)
  override protected[torch] val nativeModule: AdaptiveAvgPool1dImpl = AdaptiveAvgPool1dImpl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  def apply(t: Tensor[ParamType]): Tensor[ParamType] = fromNative(
    nativeModule.forward(t.native)
  )
  def forward(input: Tensor[ParamType]): Tensor[ParamType] = fromNative(
    nativeModule.forward(input.native)
  )

  override def toString =
    s"${getClass.getSimpleName}(outputSize=$output_size)"

}

object AdaptiveAvgPool1d:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      output_size: Int | Option[Int] | (Option[Int], Option[Int]) | (Int, Int)
  ): AdaptiveAvgPool1d[ParamType] =
    new AdaptiveAvgPool1d(output_size)

//private def nativeOutputSize = outputSize match
//  case (h: Int, w: Int) =>
//    toNative(h, w) // new LongOptionalVector(new LongOptional(h), new LongOptional(w))
//  case x: Int => toNative(x) // new LongOptionalVector(new LongOptional(x), new LongOptional(x))
//  // We know this can only be int so we can suppress the type test for Option[Int] cannot be checked at runtime warning
//  case (h: Option[Int @unchecked], w: Option[Int @unchecked]) =>
//    new LongOptionalVector(h.toOptional, w.toOptional)
//  case x: Option[Int] =>
//    new LongOptionalVector(x.toOptional, x.toOptional)
