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

// cSpell:ignore nn, inplace

package torch
package nn
package modules
package regularization

import org.bytedeco.javacpp.LongPointer
import org.bytedeco.pytorch
import org.bytedeco.pytorch.UpsampleImpl
import org.bytedeco.pytorch.{
  UpsampleOptions,
  DoubleVector,
  kNearest,
  kLinear,
  kBilinear,
  kBicubic,
  kTrilinear
}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn.modules.regularization.Upsample.UpsampleMode

// format: off
/** During training, randomly zeroes some of the elements of the input tensor with probability `p`
 * using samples from a Bernoulli distribution. Each channel will be zeroed out independently on 
 * every forward call.
 *
 * This has proven to be an effective technique for regularization and preventing the co-adaptation 
 * of neurons as described in the paper [[https://arxiv.org/abs/1207.0580 Improving neural networks 
 * by preventing co-adaptation of feature detectors]].
 *
 * Furthermore, the outputs are scaled by a factor of $\frac{1}{1−p}​ during training. This means 
 * that during evaluation the module simply computes an identity function.
 *
 * Shape:
 * - Input: $(∗)(∗)$. Input can be of any shape
 * - Output: $(∗)(∗)$. Output is of the same shape as input
 *
 * @example
 *
 * ```scala
 * import torch.nn
 *
 * val m = nn.Dropout(p=0.2)
 * val input = torch.randn(20, 16)
 * val output = m(input)
 * ```
 *
 * @param p – probability of an element to be zeroed. Default: 0.5
 * @param inplace – If set to True, will do this operation in-place. Default: `false`
 *
 * @see See [[https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_embedding.html#class-embedding Pytorch C++ Embedding]]
 * @see See [[https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#dropout Pytorch Python Dropout]]
 * @see See [[https://pytorch.org/docs/master/nn.html#torch.nn.Dropout]]
 * @see See [[https://pytorch.org/docs/master/nn.html#torch.nn.Dropout2d]]
 * @see See [[https://pytorch.org/docs/master/nn.html#torch.nn.Dropout3d]]
 * @see See [[https://pytorch.org/docs/stable/generated/torch.nn.functional.dropout.html#torch-nn-functional-dropout]]
 *
 * https://pytorch.org/docs/master/nn.html#torch.nn.Dropout
 * Add 2D, 3D, Alpha and feature alpha versions
 */
// format: on
final class Upsample[ParamType <: FloatNN | ComplexNN: Default](
    val size: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
      Option[(Int, Int, Int)] = None,
    val scale_factor: Float | (Float, Float) | (Float, Float, Float) | Option[Float] |
      Option[(Float, Float)] | Option[(Float, Float, Float)],
    val mode: UpsampleMode | String = UpsampleMode.kNearest,
    val align_corners: Option[Boolean] = None,
    val recompute_scale_factor: Option[Boolean] = None
) extends HasParams[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options: UpsampleOptions = UpsampleOptions()
  size match {
    case s: Int =>
      options.size().put(LongPointer(1).put(s.toLong))
    case s: (Int, Int)      => options.size().put(toNative(s))
    case s: (Int, Int, Int) => options.size().put(toNative(s))
    case s: Option[Int] => if s.isDefined then options.size().put(LongPointer(1).put(s.get.toLong))
    case s: Option[(Int, Int)]      => if s.isDefined then options.size().put(toNative(s.get))
    case s: Option[(Int, Int, Int)] => if s.isDefined then options.size().put(toNative(s.get))
    case None                       => ""
  }

  scale_factor match {
    case sf: Float => options.scale_factor().put(DoubleVector(1).put(sf.toDouble))
    case sf: (Float, Float) =>
      options.scale_factor().put(DoubleVector(sf._1.toDouble, sf._2.toDouble))
    case sf: (Float, Float, Float) =>
      options.scale_factor().put(DoubleVector(sf._1.toDouble, sf._2.toDouble, sf._3.toDouble))
    case sf: Option[Float] =>
      if sf.isDefined then
        options.scale_factor().put(DoubleVector(sf.get.toDouble)) // else DoubleVector(0)
    case sf: Option[(Float, Float)] =>
      if sf.isDefined then
        options
          .scale_factor()
          .put(DoubleVector(sf.get._1.toDouble, sf.get._2.toDouble)) // else DoubleVector(0)
    case sf: Option[(Float, Float, Float)] =>
      if sf.isDefined then
        options
          .scale_factor()
          .put(
            DoubleVector(sf.get._1.toDouble, sf.get._2.toDouble, sf.get._3.toDouble)
          ) // else DoubleVector(0)

  }

  if align_corners.isDefined then options.align_corners().put(align_corners.get)

  private val upsampleModeNative = mode match
    case UpsampleMode.kNearest | "nearest" | "Nearest" => options.mode().put(new kNearest)
    case UpsampleMode.kLinear | "linear" | "Linear"    => options.mode().put(new kLinear)
    case UpsampleMode.kBilinear | "bilinear" | "Bilinear" | "BiLinear" =>
      options.mode().put(new kBilinear)
    case UpsampleMode.kBicubic | "bicubic" | "Bicubic" | "BiCubic" =>
      options.mode().put(new kBicubic)
    case UpsampleMode.kTrilinear | "trilinear" | "Trilinear" | "TriLinear" =>
      options.mode().put(new kTrilinear)

  override private[torch] val nativeModule: UpsampleImpl = UpsampleImpl(options)
  nativeModule.to(paramType.toScalarType, false)

  def apply(t: Tensor[ParamType]): Tensor[ParamType] = fromNative(nativeModule.forward(t.native))

  def forward(input: Tensor[ParamType]): Tensor[ParamType] = fromNative(
    nativeModule.forward(input.native)
  )
  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  override def toString(): String =
    s"${getClass().getSimpleName()} size = ${size} scaleFactor = ${scale_factor} upsampleMode = ${mode
        .toString()} alignCorners= ${align_corners})"

object Upsample:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      size: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
        Option[(Int, Int, Int)] = None,
      scale_factor: Float | (Float, Float) | (Float, Float, Float) | Option[Float] |
        Option[(Float, Float)] | Option[(Float, Float, Float)],
      mode: UpsampleMode | String = UpsampleMode.kNearest,
      align_corners: Option[Boolean] = None,
      recompute_scale_factor: Option[Boolean] = None
  ): Upsample[ParamType] =
    new Upsample(size, scale_factor, mode, align_corners, recompute_scale_factor)
  enum UpsampleMode:
    case kNearest, kLinear, kBilinear, kBicubic, kTrilinear

//  options.mode().put(upsampleModeNative)
//  if scaleFactor != None then options.scale_factor().put(scaleFactorVec)
//  options.recompute_scale_factor().put(recomputeScaleFactor)
