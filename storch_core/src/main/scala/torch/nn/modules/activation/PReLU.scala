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
import org.bytedeco.pytorch.{PReLUImpl, PReLUOptions}
import torch.internal.NativeConverters.fromNative

/** Applies the rectified linear unit function element-wise: class torch.nn.PReLU(num_parameters=1,
  * init=0.25, device=None, dtype=None $\text{ReLU}(x) = (x)^+ = \max(0, x)$
  */
final class PReLU[D <: DType: Default](
    val num_parameters: Int = 1,
    val init: Float = 0.25f,
    val size: Option[Int] = None
) extends TensorModule[D]:

  private val options = if size.isDefined then new PReLUOptions(size.get) else new PReLUOptions()
  options.init.put(init.toDouble)
  options.num_parameters().put(num_parameters.toLong)

  override protected[torch] val nativeModule: PReLUImpl = PReLUImpl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))

  def forward(input: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(input.native))

  override def toString =
    getClass().getSimpleName() + s"(init=$init,numParameters=$num_parameters,Size=$size)"

object PReLU:
  def apply[D <: DType: Default](
      num_parameters: Int = 1,
      init: Float = 0.25f,
      size: Option[Int] = None
  ): PReLU[D] =
    new PReLU[D](num_parameters, init, size)

//  def apply[D <: DType: Default](init: Double,numParameters: Long): PReLU[D] = new PReLU[D](init,numParameters,numParameters)
//  def apply[D <: DType: Default](init: Double): PReLU[D] = new PReLU[D](init,1,1)
//  def apply[D <: DType: Default](): PReLU[D] = new PReLU[D](0.25,1,1)
