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
import org.bytedeco.pytorch.{ReLU6Impl, ReLU6Options}
import torch.internal.NativeConverters.fromNative

/** Applies the rectified linear unit function element-wise:
  *
  * $\text{ReLU}(x) = (x)^+ = \max(0, x)$
  */
final class ReLU6[D <: DType: Default](val inplace: Boolean = false, val size: Option[Int] = None)
    extends TensorModule[D]:

  private val options = if size.isDefined then ReLU6Options(size.get) else new ReLU6Options()
  options.inplace().put(inplace)

  override protected[torch] val nativeModule: ReLU6Impl = ReLU6Impl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))

  def forward(input: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(input.native))

  override def toString = getClass().getSimpleName() + s"(inplace=$inplace)"

object ReLU6:
  def apply[D <: DType: Default](inplace: Boolean = false, size: Option[Int] = None): ReLU6[D] =
    new ReLU6(inplace, size)
