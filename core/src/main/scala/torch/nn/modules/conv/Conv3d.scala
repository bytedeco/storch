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
package conv

import org.bytedeco.javacpp.{LongPointer, DoublePointer, BoolPointer}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{Conv3dImpl, Conv3dOptions, kZeros, kReflect, kReplicate, kCircular}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn.modules.conv.Conv3d.PaddingMode

/** Applies a 3D convolution over an input signal composed of several input planes.
  */
final class Conv3d[ParamType <: FloatNN | ComplexNN: Default](
    val in_channels: Int,
    val out_channels: Int,
    val kernel_size: Int | (Int, Int) | (Int, Int, Int),
    val stride: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
      Option[(Int, Int, Int)] = 1,
    val padding: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
      Option[(Int, Int, Int)] = 0,
    val dilation: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
      Option[(Int, Int, Int)] = 1,
    val groups: Int | Option[Int] = 1,
    val bias: Boolean | Option[Boolean] = true,
    val padding_mode: PaddingMode | String | Option[String] = PaddingMode.Zeros
) extends HasParams[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options =
    new Conv3dOptions(in_channels.toLong, out_channels.toLong, toNative(kernel_size))

  stride match {
    case s: Int             => options.stride().put(Array(s.toLong, s.toLong, s.toLong)*)
    case s: (Int, Int)      => options.stride().put(Array(s._1.toLong, s._2.toLong, s._2.toLong)*)
    case s: (Int, Int, Int) => options.stride().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
    case s: Option[Int] =>
      if s.isDefined then options.stride().put(Array(s.get.toLong, s.get.toLong, s.get.toLong)*)
    case s: Option[(Int, Int)] =>
      if s.isDefined then
        options.stride().put(Array(s.get._1.toLong, s.get._2.toLong, s.get._2.toLong)*)
    case s: Option[(Int, Int, Int)] =>
      if s.isDefined then
        options.stride().put(Array(s.get._1.toLong, s.get._2.toLong, s.get._3.toLong)*)
  }

  dilation match {
    case s: Int             => options.dilation().put(Array(s.toLong, s.toLong, s.toLong)*)
    case s: (Int, Int)      => options.dilation().put(Array(s._1.toLong, s._2.toLong, s._2.toLong)*)
    case s: (Int, Int, Int) => options.dilation().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
    case s: Option[Int] =>
      if s.isDefined then options.dilation().put(Array(s.get.toLong, s.get.toLong, s.get.toLong)*)
    case s: Option[(Int, Int)] =>
      if s.isDefined then
        options.dilation().put(Array(s.get._1.toLong, s.get._2.toLong, s.get._2.toLong)*)
    case s: Option[(Int, Int, Int)] =>
      if s.isDefined then
        options.dilation().put(Array(s.get._1.toLong, s.get._2.toLong, s.get._3.toLong)*)
  }

  kernel_size match {
    case s: Int        => options.kernel_size().put(Array(s.toLong, s.toLong, s.toLong)*)
    case s: (Int, Int) => options.kernel_size().put(Array(s._1.toLong, s._2.toLong, s._2.toLong)*)
    case s: (Int, Int, Int) =>
      options.kernel_size().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
  }

  padding match {
    case s: Int             => options.padding().put(toNative((s, s, s)))
    case s: (Int, Int)      => options.padding().put(toNative(s))
    case s: (Int, Int, Int) => options.padding().put(toNative(s))
    case s: Option[Int] =>
      if s.isDefined then options.padding().put(toNative((s.get, s.get, s.get)))
    case s: Option[(Int, Int)]      => if s.isDefined then options.padding().put(toNative(s.get))
    case s: Option[(Int, Int, Int)] => if s.isDefined then options.padding().put(toNative(s.get))
  }

  groups match {
    case g: Int         => options.groups().put(g)
    case g: Option[Int] => if g.isDefined then options.groups().put(g.get)
  }
  bias match {
    case b: Boolean         => options.bias().put(b)
    case b: Option[Boolean] => if b.isDefined then options.bias().put(b.get)
  }

  padding_mode match
    case PaddingMode.Zeros | "zeros" | "Zeros" | Some("zeros") | Some("Zeros") =>
      options.padding_mode().put(new kZeros)
    case PaddingMode.Reflect | "reflect" | "Reflect" | Some("reflect") | Some("Reflect") =>
      options.padding_mode().put(new kReflect)
    case PaddingMode.Replicate | "replicate" | "Replicate" | Some("replicate") | Some(
          "Replicate"
        ) =>
      options.padding_mode().put(new kReplicate)
    case PaddingMode.Circular | "circular" | "Circular" | Some("cirular") | Some("Cirular") =>
      options.padding_mode().put(new kCircular)

  override private[torch] val nativeModule: Conv3dImpl = Conv3dImpl(options)
  nativeModule.to(paramType.toScalarType, false)

  def apply(t: Tensor[ParamType]): Tensor[ParamType] = fromNative(nativeModule.forward(t.native))

  def forward(input: Tensor[ParamType]): Tensor[ParamType] = fromNative(
    nativeModule.forward(input.native)
  )

  def weight: Tensor[ParamType] = fromNative(nativeModule.weight)

  override def hasBias(): Boolean = options.bias().get()

  def bias_(): Tensor[ParamType] = fromNative(nativeModule.bias)

  def bias(un_used: Int*): Tensor[ParamType] = fromNative(nativeModule.bias)

  def reset(): Unit = nativeModule.reset()

  def reset_parameters(): Unit = nativeModule.reset_parameters()

  override def toString =
    s"${getClass.getSimpleName} (in_channels $in_channels, out_channels $out_channels, kernel_size=$kernel_size, stride=$stride, padding=$padding, bias=$bias)"

object Conv3d:
  def apply[T <: FloatNN | ComplexNN: Default](
      in_channels: Int,
      out_channels: Int,
      kernel_size: Int | (Int, Int) | (Int, Int, Int),
      stride: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
        Option[(Int, Int, Int)] = 1,
      padding: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
        Option[(Int, Int, Int)] = 0,
      dilation: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
        Option[(Int, Int, Int)] = 1,
      groups: Int | Option[Int] = 1,
      bias: Boolean | Option[Boolean] = true,
      padding_mode: PaddingMode | String | Option[String] = PaddingMode.Zeros
  ): Conv3d[T] =
    new Conv3d[T](
      in_channels,
      out_channels,
      kernel_size,
      stride,
      padding,
      dilation,
      groups,
      bias,
      padding_mode
    )
  enum PaddingMode:
    case Zeros, Reflect, Replicate, Circular

//  stride match {
//    case s: Int => options.stride().put(toNative((s,s,s)))
//    case s: (Int, Int) => options.stride().put(toNative(s))
//    case s: (Int, Int, Int) => options.stride().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
//    case s: Option[Int] => if s.isDefined then options.stride().put(toNative((s.get,s.get,s.get)))
//    case s: Option[(Int, Int)] => if s.isDefined then options.stride().put(toNative(s.get))
//    case s: Option[(Int, Int, Int)] => if s.isDefined then options.stride().put(Array(s.get._1.toLong, s.get._2.toLong, s.get._3.toLong)*)
//  }
//  padding match {
//    case s: Int => options.padding().put(Array(s.toLong, s.toLong, s.toLong) *)
//    case s: (Int, Int) => options.padding().put(Array(s._1.toLong, s._2.toLong, s._2.toLong) *)
//    case s: (Int, Int, Int) => options.padding().put(Array(s._1.toLong, s._2.toLong, s._3.toLong) *)
//    case s: Option[Int] => if s.isDefined then options.padding().put(Array(s.get.toLong, s.get.toLong, s.get.toLong) *)
//    case s: Option[(Int, Int)] => if s.isDefined then options.padding().put(Array(s.get._1.toLong, s.get._2.toLong, s.get._2.toLong) *)
//    case s: Option[(Int, Int, Int)] => if s.isDefined then options.padding().put(Array(s.get._1.toLong, s.get._2.toLong, s.get._3.toLong) *)
//
//  }
//  dilation match {
//    case s: Int => options.dilation().put(toNative((s,s,s)))
//    case s: (Int, Int) => options.dilation().put(toNative(s))
//    case s: (Int, Int, Int) => options.dilation().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
//    case s: Option[Int] => if s.isDefined then options.dilation().put(toNative((s.get,s.get,s.get)))
//    case s: Option[(Int, Int)] => if s.isDefined then options.dilation().put(toNative(s.get))
//    case s: Option[(Int, Int, Int)] => if s.isDefined then options.dilation().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
//  }
//  options.stride().put(toNative(stride))
//  options.padding().put(toNative(padding))
//  options.dilation().put(toNative(dilation))
//  options.groups().put(groups)
//  options.bias().put(bias)
//  private val paddingModeNative =
//  options.stride().put(toNative(stride))
//  options.padding().put(toNative(padding))
//  options.dilation().put(toNative(dilation))
//  options.groups().put(groups)
//  options.bias().put(bias)
//  options.padding_mode().put()

//  private val paddingModeNative = paddingMode match
//    case PaddingMode.Zeros     => new kZeros
//    case PaddingMode.Reflect   => new kReflect
//    case PaddingMode.Replicate => new kReplicate
//    case PaddingMode.Circular  => new kCircular
//
//  private val paddingModeNative = paddingMode match
//    case PaddingMode.Zeros | "zeros" | "Zeros" => new kZeros
//    case PaddingMode.Reflect | "reflect" | "Reflect" => new kReflect
//    case PaddingMode.Replicate | "replicate" | "Replicate" => new kReplicate
//    case PaddingMode.Circular | "circular" | "Circular" => new kCircular
//  options.padding_mode().put(paddingModeNative)
