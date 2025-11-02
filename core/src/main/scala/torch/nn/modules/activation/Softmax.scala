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
import org.bytedeco.pytorch.SoftmaxImpl
import org.bytedeco.pytorch.SoftmaxOptions
import torch.internal.NativeConverters.fromNative

/** Applies the Softmax function to an n-dimensional input Tensor rescaling them so that the
  * elements of the n-dimensional output Tensor lie in the range [0,1] and sum to 1.
  *
  * Softmax is defined as: $$\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$ class
  * torch.nn.Softmax(dim=None)[source] >>> m = nn.Softmax(dim=1) >>> input = torch.randn(2, 3) >>>
  * output = m(input) >>> print(output) tensor([[0.2689, 0.6810, 0.0501], [0.2689, 0.6810, 0.0501]])
  * (x$0: Long): org.bytedeco.pytorch.SoftmaxOptions [error] | (x$0: org.bytedeco.javacpp.Pointer):
  * org.bytedeco.pytorch.SoftmaxOptions When the input Tensor is a sparse tensor then the unspecifed
  * values are treated as ``-inf``.
  */
final class Softmax[D <: DType: Default](val dim: Int = 1) extends TensorModule[D]:

  private val options = new SoftmaxOptions(dim)

  override val nativeModule: SoftmaxImpl = SoftmaxImpl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))
  def forward(input: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(input.native))

  override def toString = getClass().getSimpleName() + s"(dim=$dim)"

object Softmax:
  def apply[D <: DType: Default](dim: Int = 1): Softmax[D] = new Softmax(dim)
