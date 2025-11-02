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

import org.bytedeco.javacpp.{BytePointer}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{UnflattenImpl, UnflattenOptions, StringLongVector, LongVector}
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
final class Unflatten[D <: DType: Default](
    val dim: Int | Option[Int] = 1,
    val unflattened_size: Seq[Int] | (Int, Int) | (Int, Int, Int) | (Int, Int, Int, Int) |
      Option[Seq[Int]] = None,
    val dim_name: Option[String] = None,
    val named_shape: Option[Map[String, Int]] = None
) extends TensorModule[D]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  val size = unflattened_size match {
    case s: (Int, Int)      => LongVector(Array(s._1.toLong, s._2.toLong)*)
    case s: (Int, Int, Int) => LongVector(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
    case s: (Int, Int, Int, Int) =>
      LongVector(Array(s._1.toLong, s._2.toLong, s._3.toLong, s._4.toLong)*)
    case s: Seq[Int] => LongVector(s.map(_.toLong)*)
    case s: Option[Seq[Int]] =>
      if s.isDefined then LongVector(s.get.map(_.toLong)*) else LongVector(Array(-1).map(_.toLong)*)
  }

  private val options = dim match {
    case d: Int => UnflattenOptions(d.toLong, size)
    case d: Option[Int] =>
      if d.isDefined then UnflattenOptions(d.get.toLong, size)
      else
        UnflattenOptions(
          dim_name.get,
          StringLongVector(
            named_shape.get.keys.toArray,
            named_shape.get.values.map(_.toLong).toArray
          )
        )

  }
  dim_name match {
    case d: Option[String] =>
      if d.isDefined then options.dimname().put(BytePointer(d.get))
  }

  named_shape match {
    case d: Option[Map[String, Int]] =>
      if d.isDefined then
        options
          .namedshape()
          .put(StringLongVector(d.get.keys.toArray, d.get.values.map(_.toLong).toArray))

  }

  override val nativeModule: UnflattenImpl = UnflattenImpl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))

  def forward(input: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(input.native))

  override def toString =
    s"${getClass.getSimpleName}(dim = ${dim}, dimName = ${dim_name} namedShape ${named_shape
        .toString()} sizes ${unflattened_size})"

object Unflatten:

  def apply[D <: DType: Default](
      dim: Int | Option[Int] = 1,
      unflattened_size: Seq[Int] | (Int, Int) | (Int, Int, Int) | (Int, Int, Int, Int) |
        Option[Seq[Int]] = None,
      dim_name: Option[String] = None,
      named_shape: Option[Map[String, Int]] = None
  ): Unflatten[D] = new Unflatten[D](dim, unflattened_size, dim_name, named_shape)

//  if dimName.isDefined then
//     if options.isDefined then
//        options.dimname().put(BytePointer(dimName.get))
//  if namedShape.isDefined then
//     if options.isDefined then
//        val namesShapeVec = StringLongVector(namedShape.get.keys.toArray, namedShape.get.values.map(_.toLong).toArray)
//        options.namedshape().put(namesShapeVec)
