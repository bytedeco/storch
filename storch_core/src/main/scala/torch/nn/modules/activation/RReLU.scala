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
import org.bytedeco.pytorch.{RReLUImpl, RReLUOptions}
import torch.internal.NativeConverters.fromNative

/** Applies the rectified linear unit function element-wise: class torch.nn.RReLU(lower=0.125,
  * upper=0.3333333333333333, inplace=False)[source] $\text{ReLU}(x) = (x)^+ = \max(0, x)$
  */
final class RReLU[D <: DType: Default](
    val lower: Float = 0.125f,
    val upper: Float = 0.3333333333333333f,
    val inplace: Boolean = false,
    val size: Option[Int] = None
) extends TensorModule[D]:

  private val options = if size.isDefined then RReLUOptions(size.get) else RReLUOptions()
  options.inplace().put(inplace)
  options.lower().put(lower.toDouble)
  options.upper().put(upper.toDouble)

  override protected[torch] val nativeModule: RReLUImpl = RReLUImpl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))

  def forward(input: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(input.native))

  override def toString =
    getClass().getSimpleName() + s"(lower=$lower,upper=$upper,inplace=$inplace)"

object RReLU:
  def apply[D <: DType: Default](
      lower: Float = 0.125f,
      upper: Float = 0.3333333333333333f,
      inplace: Boolean = false,
      size: Option[Int] = None
  ): RReLU[D] =
    new RReLU[D](lower, upper, inplace, size)
