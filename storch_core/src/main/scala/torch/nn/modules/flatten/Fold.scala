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
import org.bytedeco.pytorch.{FoldImpl, FoldOptions}
import torch.internal.NativeConverters.{fromNative, toNative}

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
final class Fold[D <: DType: Default](
    val output_size: Int | (Int, Int) | (Int, Int, Int),
    val kernel_size: Int | (Int, Int) | (Int, Int, Int),
    val dilation: Int | (Int, Int) | Option[Int] | Option[(Int, Int)] = 1,
    val padding: Int | (Int, Int) | Option[Int] | Option[(Int, Int)] = 0,
    val stride: Int | (Int, Int) = 1
) extends TensorModule[D]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options = FoldOptions(toNative(output_size), toNative(kernel_size))

  dilation match {
    case d: Int                => options.dilation().put(toNative(d))
    case d: (Int, Int)         => options.dilation().put(toNative(d))
    case d: Option[Int]        => if d.isDefined then options.dilation().put(toNative(d.get))
    case d: Option[(Int, Int)] => if d.isDefined then options.dilation().put(toNative(d.get))
  }
  padding match {
    case p: Int                => options.padding().put(toNative(p))
    case p: (Int, Int)         => options.padding().put(toNative(p))
    case p: Option[Int]        => if p.isDefined then options.padding().put(toNative(p.get))
    case p: Option[(Int, Int)] => if p.isDefined then options.padding().put(toNative(p.get))
  }

  stride match {
    case s: Int        => options.stride().put(Array(s.toLong)*)
    case s: (Int, Int) => options.stride().put(Array(s._1.toLong, s._2.toLong)*)
  }

  output_size match {
    case s: Int        => options.output_size().put(Array(s.toLong)*)
    case s: (Int, Int) => options.output_size().put(Array(s._1.toLong, s._2.toLong)*)
    case s: (Int, Int, Int) =>
      options.output_size().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
  }

  kernel_size match {
    case s: Int        => options.kernel_size().put(Array(s.toLong)*)
    case s: (Int, Int) => options.kernel_size().put(Array(s._1.toLong, s._2.toLong)*)
    case s: (Int, Int, Int) =>
      options.kernel_size().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
  }

  override val nativeModule: FoldImpl = FoldImpl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))

  def forward(input: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(input.native))

  override def toString =
    s"${getClass.getSimpleName}(outputSize = ${output_size}, kernelSize = ${kernel_size}, dilation = ${dilation}, padding = ${padding}, stride = ${stride}"

object Fold:
  def apply[D <: DType: Default](
      output_size: Int | (Int, Int) | (Int, Int, Int),
      kernel_size: Int | (Int, Int) | (Int, Int, Int),
      dilation: Int | (Int, Int) | Option[Int] | Option[(Int, Int)] = 1,
      padding: Int | (Int, Int) | Option[Int] | Option[(Int, Int)] = 0,
      stride: Int | (Int, Int) = 1
  ): Fold[D] = new Fold[D](output_size, kernel_size, dilation, padding, stride)

//  options.stride().put(toNative(stride))
//  options.output_size().put(toNative(outputSize))
//  options.kernel_size().put(toNative(kernelSize))
