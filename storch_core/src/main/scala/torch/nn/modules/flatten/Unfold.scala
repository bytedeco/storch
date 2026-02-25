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
package flatten

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{UnfoldImpl, UnfoldOptions}
import torch.internal.NativeConverters.{toNative, fromNative}
// format: off
/** Flattens a contiguous range of dims into a tensor. For use with [[nn.Sequential]].
 *
 * Shape:
 * \- Input: $(*, S_{\text{start}},..., S_{i}, ..., S_{\text{end}}, *)$,' where $S_{i}$ is the size
 * at dimension $i$ and $*$ means any number of dimensions including none.
 * \- Output: $(*, \prod_{i=\text{start}}^{\text{end}} S_{i}, *)$.
 *
 * Example:
 *
 * ```scala
 * import torch.nn
 *
 * val input = torch.randn(Seq(32, 1, 5, 5))
 * // With default parameters
 * val m1 = nn.Flatten()
 * // With non-default parameters
 * val m2 = nn.Flatten(0, 2)
 * ```
 *
 * @group nn_flatten
 *
 * @param startDim
 *   first dim to flatten
 * @param endDim
 *   last dim to flatten
 */
// format: on
final class Unfold[D <: DType: Default](
    val kernel_size: Int | (Int, Int) | (Int, Int, Int),
    val dilation: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
      Option[(Int, Int, Int)] = 1,
    val padding: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
      Option[(Int, Int, Int)] = 0,
    val stride: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
      Option[(Int, Int, Int)] = 1
) extends TensorModule[D]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options = UnfoldOptions(toNative(kernel_size))
  options.kernel_size().put(toNative(kernel_size))
  stride match {
    case s: Int                     => options.stride().put(toNative(s))
    case s: (Int, Int)              => options.stride().put(toNative(s))
    case s: (Int, Int, Int)         => options.stride().put(toNative(s))
    case s: Option[Int]             => if s.isDefined then options.stride().put(toNative(s.get))
    case s: Option[(Int, Int)]      => if s.isDefined then options.stride().put(toNative(s.get))
    case s: Option[(Int, Int, Int)] => if s.isDefined then options.stride().put(toNative(s.get))
  }
  padding match {
    case s: Int                     => options.padding().put(toNative(s))
    case s: (Int, Int)              => options.padding().put(toNative(s))
    case s: (Int, Int, Int)         => options.padding().put(toNative(s))
    case s: Option[Int]             => if s.isDefined then options.padding().put(toNative(s.get))
    case s: Option[(Int, Int)]      => if s.isDefined then options.padding().put(toNative(s.get))
    case s: Option[(Int, Int, Int)] => if s.isDefined then options.padding().put(toNative(s.get))
  }
  dilation match {
    case s: Int                     => options.dilation().put(toNative(s))
    case s: (Int, Int)              => options.dilation().put(toNative(s))
    case s: (Int, Int, Int)         => options.dilation().put(toNative(s))
    case s: Option[Int]             => if s.isDefined then options.dilation().put(toNative(s.get))
    case s: Option[(Int, Int)]      => if s.isDefined then options.dilation().put(toNative(s.get))
    case s: Option[(Int, Int, Int)] => if s.isDefined then options.dilation().put(toNative(s.get))
  }

  override val nativeModule: UnfoldImpl = UnfoldImpl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))
  def forward(input: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(input.native))

  override def toString =
    s"${getClass.getSimpleName}( kernelSize = ${kernel_size}, dilation = ${dilation}, padding = ${padding}, stride = ${stride}"

object Unfold:

  def apply[D <: DType: Default](
      kernel_size: Int | (Int, Int) | (Int, Int, Int),
      dilation: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
        Option[(Int, Int, Int)] = 1,
      padding: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
        Option[(Int, Int, Int)] = 0,
      stride: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
        Option[(Int, Int, Int)] = 1
  ): Unfold[D] = new Unfold(kernel_size, dilation, padding, stride)
