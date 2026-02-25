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
package activation

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{ELUOptions, SELUImpl, SELUOptions, SoftmaxOptions}
import torch.internal.NativeConverters.fromNative

/** Applies the Softmax function to an n-dimensional input Tensor rescaling them so that the
  * elements of the n-dimensional output Tensor lie in the range [0,1] and sum to 1.
  *
  * Softmax is defined as: $$\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$
  *
  * When the input Tensor is a sparse tensor then the unspecifed values are treated as ``-inf``.
  */
final class SELU[D <: DType: Default](val inplace: Boolean = false, val size: Option[Int] = None)
    extends TensorModule[D]:

  val options = if size.isDefined then SELUOptions(size.get) else SELUOptions()
  options.inplace().put(inplace)
  override val nativeModule: SELUImpl = SELUImpl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))

  def forward(input: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(input.native))

  override def toString = getClass().getSimpleName() + s"(size=$size,inplace=$inplace)"

object SELU:
  def apply[D <: DType: Default](inplace: Boolean = false, size: Option[Int] = None): SELU[D] =
    new SELU(inplace, size)
