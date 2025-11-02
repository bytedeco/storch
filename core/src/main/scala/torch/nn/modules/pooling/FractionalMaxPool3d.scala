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
package pooling

import org.bytedeco.javacpp.{LongPointer, DoublePointer}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{FractionalMaxPool3dImpl, FractionalMaxPool3dOptions}
import torch.internal.NativeConverters.{fromNative, toNative}

/** Applies a 2D max pooling over an input signal composed of several input planes.
  * T_TensorTensor_T, LongExpandingArrayOptional, DoubleExpandingArrayOptional,
  */
final class FractionalMaxPool3d[D <: FloatNN | ComplexNN: Default](
    val kernel_size: Int | (Int, Int, Int),
    val output_size: Option[Int] | Option[(Int, Int, Int)] = None,
    val output_ratio: Option[Float] | Option[(Float, Float, Float)] = None,
    val return_indices: Boolean = false,
    val random_samples: Option[Seq[Float] | Tensor[D]] = None
) extends TensorModule[D]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options: FractionalMaxPool3dOptions = FractionalMaxPool3dOptions(
    toNative(kernel_size)
  )

  if output_size.isDefined then
    output_size.get match {
      case t: Int =>
//      options.output_size().put(Array(t.toLong,t.toLong)*)
        options.output_size().put(LongPointer(t.toLong))
        options.output_size().put(LongPointer(t.toLong))
        println("output_size 3 same elements full")
      //      options.output_size().put(LongPointer(Array(t.toLong,t.toLong)*))
      case t: (Int, Int, Int) =>
        options.output_size().put(LongPointer(t._1))
        options.output_size().put(LongPointer(t._2))
        options.output_size().put(LongPointer(t._3))
        println("output_size three elements full")
      //      options.output_size().put(toNative(t))
    }
  if output_ratio.isDefined then
    output_ratio.get match {
      case t: Float =>
        options.output_ratio().put(DoublePointer(t.toDouble))
        options.output_ratio().put(DoublePointer(t.toDouble))
        options.output_ratio().put(DoublePointer(t.toDouble))
        println(s"output ratio 3 same elements full")
      //      options.output_ratio().put(DoublePointer(Array(t.toDouble,t.toDouble)*))
      case t: (Float, Float, Float) =>
        options.output_ratio().put(DoublePointer(t._1.toDouble))
        options.output_ratio().put(DoublePointer(t._2.toDouble))
        options.output_ratio().put(DoublePointer(t._3.toDouble))
        println(s"output ratio three  elements full")
      //      options.output_ratio().put(DoublePointer(Array(t._1.toDouble, t._2.toDouble)*))
    }

  options.kernel_size().put(toNative(kernel_size))
  random_samples match {
    case Some(t: Seq[Float]) => options._random_samples().put(torch.Tensor(t).native)
    case Some(t: Tensor[D])  => options._random_samples().put(t.native)
    case None =>
      println(s"randomSamples is None outputSize ${output_size} outputRatio ${output_ratio}")
  }
  println(s"FractionalMaxPool3d raw  options kernel ${options.kernel_size().get(0)} k2 ${options
      .kernel_size()
      .get(1)}  k3 ${options.kernel_size().get(2)} outsize ${options.output_size().has_value()}  ${options
      .output_size()
      .get()
      .get(0)} out2 ${options.output_size().get().get(1)} out3 ${options.output_size().get().get(2)} outRatio ${options
      .output_ratio()
      .has_value()} ${options.output_ratio().get().get(0)} ratio2 ${options
      .output_ratio()
      .get()
      .get(1)}  ratio3 ${options.output_ratio().get().get(2)}")

  override private[torch] val nativeModule: FractionalMaxPool3dImpl = FractionalMaxPool3dImpl(
    options
  )
  println(s"FractionalMaxPool3d  options kernel ${nativeModule.options().kernel_size().get(0)} k2 ${nativeModule
      .options()
      .kernel_size()
      .get(1)}   k3 ${options.kernel_size().get(2)}  outsize ${nativeModule.options().output_size().has_value()}  ${nativeModule
      .options()
      .output_size()
      .get()
      .get(0)} out2 ${nativeModule.options().output_size().get().get(1)} outRatio ${nativeModule
      .options()
      .output_ratio()
      .has_value()} ${nativeModule.options().output_ratio().get().get(0)} ratio2 ${nativeModule.options().output_ratio().get().get(1)}")

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  override def toString(): String =
    s"${getClass.getSimpleName}(kernelSize=$kernel_size,outputSize ${output_size} outputRatio ${output_ratio} returnIndices ${return_indices} randomSamples ${random_samples} )"

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))
  def forward(input: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(input.native))

  def forward_with_indices(t: Tensor[D]): (Tensor[D], Tensor[D]) =
    val outputWithIndices = nativeModule.forward_with_indices(t.native)
    (fromNative(outputWithIndices.get0()), fromNative(outputWithIndices.get1()))

object FractionalMaxPool3d:
  def apply[D <: FloatNN | ComplexNN: Default](
      kernel_size: Int | (Int, Int, Int),
      output_size: Option[Int] | Option[(Int, Int, Int)],
      output_ratio: Option[Float] | Option[(Float, Float, Float)],
      return_indices: Boolean = false,
      random_samples: Option[Seq[Float] | Tensor[D]] = None
  ): FractionalMaxPool3d[D] =
    new FractionalMaxPool3d(
      kernel_size,
      output_size,
      output_ratio,
      return_indices,
      random_samples
    )

//  println(s"doubleArrayOptionalOutputRatio set after has_value ${doubleArrayOptionalOutputRatio.has_value()}")
//
//  println(s"longArrayOptionalOutputSize set after has_value ${longArrayOptionalOutputSize.has_value()}")

//  if outputRatio.isDefined then outputRatio.get match {
//    case t: Float => options.output_ratio().put(DoublePointer(Array(t.toDouble,t.toDouble,t.toDouble)*))
//    case t: (Float, Float,Float) => options.output_ratio().put(DoublePointer(Array(t._1.toDouble, t._2.toDouble) *))
//  }

//  options.output_size().put(longArrayOptionalOutputSize)

//  options.output_ratio().put(doubleArrayOptionalOutputRatio)

//  val longArrayOptionalOutputSize: LongExpandingArrayOptional = new LongExpandingArrayOptional()
//  val doubleArrayOptionalOutputRatio: DoubleExpandingArrayOptional = new DoubleExpandingArrayOptional()
//  println(s"doubleArrayOptionalOutputRatio init has_value ${doubleArrayOptionalOutputRatio.has_value()}")
//
//  println(s"longArrayOptionalOutputSize init has_value ${longArrayOptionalOutputSize.has_value()}")
//  if outputSize.isDefined then outputSize.get match {
//    case t: Int => longArrayOptionalOutputSize.put(toNative(t))
//    case t: (Int,Int,Int) => longArrayOptionalOutputSize.put(toNative(t))
//
//  }
//  if outputRatio.isDefined then  outputRatio.get  match {
//    case t: Float => doubleArrayOptionalOutputRatio.put(DoublePointer(t.toDouble))
//    case t: (Float, Float, Float) => doubleArrayOptionalOutputRatio.put(DoublePointer(Array(t._1.toDouble, t._2.toDouble, t._3.toDouble)*))
//
//  }

//  if outputSize.isDefined then outputSize.get match {
//    case t: Int => options.output_size().put(LongPointer(Array(t.toLong,t.toLong,t.toLong)*))
//    case t: (Int, Int,Int) => options.output_size().put(toNative(t))
//  }
