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
import org.bytedeco.pytorch.{HardtanhImpl, HardtanhOptions}
import torch.internal.NativeConverters.fromNative

/** Applies the rectified linear unit function element-wise:
  *
  * $\text{ReLU}(x) = (x)^+ = \max(0, x)$
  */
final class Hardtanh[D <: DType: Default](
    val min_val: Float,
    val max_val: Float,
    val inplace: Boolean = false,
    val size: Option[Int] = None
) extends TensorModule[D]:

  private val options =
    if size.isDefined then new HardtanhOptions(size.get) else new HardtanhOptions()
  options.inplace().put(inplace)
  options.min_val().put(min_val.toDouble)
  options.max_val().put(max_val.toDouble)

  override protected[torch] val nativeModule: HardtanhImpl = HardtanhImpl(options)

  override def hasBias(): Boolean = false

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))

  def forward(input: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(input.native))

  override def toString =
    getClass().getSimpleName() + s"(size=$size,minVal=$min_val,maxVal=$max_val,inplace=$inplace)"

object Hardtanh:
  def apply[D <: DType: Default](
      min_val: Float,
      max_val: Float,
      inplace: Boolean = false,
      size: Option[Int] = None
  ): Hardtanh[D] = new Hardtanh(min_val, max_val, inplace, size)
