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

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ConvTranspose1dImpl,
  LongArrayRefOptional,
  ConvTranspose1dOptions,
  kZeros,
  kReflect,
  kReplicate,
  kCircular
}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn.modules.conv.ConvTranspose1d.PaddingMode

/** Applies a 2D convolution over an input signal composed of several input planes.
  *
  * @group nn_conv
  */
final class ConvTranspose1d[ParamType <: FloatNN | ComplexNN: Default](
    val in_channels: Int,
    val out_channels: Int,
    val kernel_size: Int | (Int, Int),
    val stride: Int | (Int, Int) | Option[Int] | Option[(Int, Int)] = 1,
    val padding: Int | (Int, Int) | Option[Int] | Option[(Int, Int)] = 0,
    val output_padding: Int | (Int, Int) | Option[Int] | Option[(Int, Int)] = 0,
    val groups: Int | Option[Int] = 1,
    val bias: Boolean | Option[Boolean] = true,
    val dilation: Int | (Int, Int) | Option[Int] | Option[(Int, Int)] = 1,
    val padding_mode: PaddingMode | String | Option[String] = PaddingMode.Zeros
) extends HasParams[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options =
    new ConvTranspose1dOptions(in_channels.toLong, out_channels.toLong, toNative(kernel_size))
  stride match {
    case s: Int                => options.stride().put(toNative(s))
    case s: (Int, Int)         => options.stride().put(toNative(s))
    case s: Option[Int]        => if s.isDefined then options.stride().put(toNative(s.get))
    case s: Option[(Int, Int)] => if s.isDefined then options.stride().put(toNative(s.get))
  }
  padding match {
    case s: Int                => options.padding().put(toNative(s))
    case s: (Int, Int)         => options.padding().put(toNative(s))
    case s: Option[Int]        => if s.isDefined then options.padding().put(toNative(s.get))
    case s: Option[(Int, Int)] => if s.isDefined then options.padding().put(toNative(s.get))
  }
  output_padding match {
    case s: Int        => options.output_padding().put(toNative((s, s)))
    case s: (Int, Int) => options.output_padding().put(toNative(s))
    case s: Option[Int] =>
      if s.isDefined then options.output_padding().put(toNative((s.get, s.get)))
    case s: Option[(Int, Int)] => if s.isDefined then options.output_padding().put(toNative(s.get))
  }
  dilation match {
    case s: Int                => options.dilation().put(toNative(s))
    case s: (Int, Int)         => options.dilation().put(toNative(s))
    case s: Option[Int]        => if s.isDefined then options.dilation().put(toNative(s.get))
    case s: Option[(Int, Int)] => if s.isDefined then options.dilation().put(toNative(s.get))
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

  override private[torch] val nativeModule: ConvTranspose1dImpl = ConvTranspose1dImpl(options)
  nativeModule.to(paramType.toScalarType, false)

  def apply(t: Tensor[ParamType]): Tensor[ParamType] = fromNative(nativeModule.forward(t.native))

  def forward(input: Tensor[ParamType]): Tensor[ParamType] = fromNative(
    nativeModule.forward(input.native)
  )

  def apply(input: Tensor[ParamType], output_size: Seq[Int]): Tensor[ParamType] = {

    val outputSize = LongArrayRefOptional(output_size.map(_.toLong)*)
    fromNative(nativeModule.forward(input.native, outputSize))
  }
  def forward(input: Tensor[ParamType], output_size: Seq[Int]): Tensor[ParamType] = {

    val outputSize = LongArrayRefOptional(output_size.map(_.toLong)*)
    fromNative(nativeModule.forward(input.native, outputSize))
  }

  def weight: Tensor[ParamType] = fromNative(nativeModule.weight)

  def bias_(): Tensor[ParamType] = fromNative(nativeModule.bias)

  def bias(un_used: Int*): Tensor[ParamType] = fromNative(nativeModule.bias)

  def reset(): Unit = nativeModule.reset()

  def reset_parameters(): Unit = nativeModule.reset_parameters()

  override def hasBias(): Boolean = options.bias().get()

  override def toString =
    s"${getClass.getSimpleName} ($in_channels, $out_channels, kernelSize=$kernel_size, stride=$stride, padding=$padding, bias=$bias)"

object ConvTranspose1d:
  def apply[T <: FloatNN | ComplexNN: Default](
      in_channels: Int,
      out_channels: Int,
      kernel_size: Int | (Int, Int),
      stride: Int | (Int, Int) | Option[Int] | Option[(Int, Int)] = 1,
      padding: Int | (Int, Int) | Option[Int] | Option[(Int, Int)] = 0,
      output_padding: Int | (Int, Int) | Option[Int] | Option[(Int, Int)] = 0,
      groups: Int | Option[Int] = 1,
      bias: Boolean | Option[Boolean] = true,
      dilation: Int | (Int, Int) | Option[Int] | Option[(Int, Int)] = 1,
      padding_mode: PaddingMode | String | Option[String] = PaddingMode.Zeros
  ): ConvTranspose1d[T] =
    new ConvTranspose1d[T](
      in_channels,
      out_channels,
      kernel_size,
      stride,
      padding,
      output_padding,
      groups,
      bias,
      dilation,
      padding_mode
    )
  enum PaddingMode:
    case Zeros, Reflect, Replicate, Circular
