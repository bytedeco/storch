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
package functional

import org.bytedeco.javacpp.LongPointer
import org.bytedeco.pytorch.{UnfoldOptions, ConvTranspose1dFuncOptions, FoldOptions}
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.*

private[torch] trait Convolution {

  /** Applies a 1D convolution over an input signal composed of several input planes.
    *
    * @group nn_conv
    */
  def conv1d[D <: FloatNN | ComplexNN](
      input: Tensor[D],
      weight: Tensor[D],
      bias: Tensor[D] | Option[Tensor[D]] = None,
      stride: Int = 1,
      padding: Int = 0,
      dilation: Int = 1,
      groups: Int = 1
  ): Tensor[D] =
    fromNative(
      torchNative.conv1d(
        input.native,
        weight.native,
        toOptional(bias),
        Array(stride.toLong),
        Array(padding.toLong),
        Array(dilation.toLong),
        groups
      )
    )

  /** Applies a 2D convolution over an input signal composed of several input planes.
    *
    * @group nn_conv
    */
  def conv2d[D <: FloatNN | ComplexNN](
      input: Tensor[D],
      weight: Tensor[D],
      bias: Tensor[D] | Option[Tensor[D]] = None,
      stride: Int | (Int, Int) = 1,
      padding: Int | (Int, Int) = 0,
      dilation: Int | (Int, Int) = 1,
      groups: Int = 1
  ): Tensor[D] =
    fromNative(
      torchNative.conv2d(
        input.native,
        weight.native,
        toOptional(bias),
        toArray(stride),
        toArray(padding),
        toArray(dilation),
        groups
      )
    )

  /** Applies a 3D convolution over an input image composed of several input planes.
    *
    * @group nn_conv
    */
  def conv3d[D <: FloatNN | ComplexNN](
      input: Tensor[D],
      weight: Tensor[D],
      bias: Tensor[D] | Option[Tensor[D]] = None,
      stride: Int = 1,
      padding: Int = 0,
      dilation: Int = 1,
      groups: Int = 1
  ): Tensor[D] =
    fromNative(
      torchNative.conv3d(
        input.native,
        weight.native,
        toOptional(bias),
        Array(stride.toLong),
        Array(padding.toLong),
        Array(dilation.toLong),
        groups
      )
    )

  /** Applies a 1D transposed convolution operator over an input signal composed of several input
    * planes, sometimes also called “deconvolution”. ConvTranspose1dFuncOptions
    * @group nn_conv
    */
  def conv_transpose1d[D <: FloatNN | ComplexNN](
      input: Tensor[D],
      weight: Tensor[D],
      bias: Tensor[D] | Option[Tensor[D]] = None,
      stride: Int | (Int, Int) = 1,
      padding: Int | (Int, Int) = 0,
      output_padding: Int | (Int, Int) = 0,
      groups: Int = 1,
      dilation: Int | (Int, Int) = 1
  ): Tensor[D] =
    convTranspose1d(input, weight, bias, stride, padding, output_padding, groups, dilation)

  def convTranspose1d[D <: FloatNN | ComplexNN](
      input: Tensor[D],
      weight: Tensor[D],
      bias: Tensor[D] | Option[Tensor[D]] = None,
      stride: Int | (Int, Int) = 1,
      padding: Int | (Int, Int) = 0,
      outputPadding: Int | (Int, Int) = 0,
      groups: Int = 1,
      dilation: Int | (Int, Int) = 1
  ): Tensor[D] =
    fromNative(
      torchNative.conv_transpose1d(
        input.native,
        weight.native,
        toOptional(bias),
        toArray(stride),
        toArray(padding),
        toArray(outputPadding),
        groups,
        toArray(dilation): _*
      )
    )

  /** Applies a 2D transposed convolution operator over an input image composed of several input
    * planes, sometimes also called “deconvolution”. ConvTranspose2dFuncOptions
    * @group nn_conv
    */
  def conv_transpose2d[D <: FloatNN | ComplexNN](
      input: Tensor[D],
      weight: Tensor[D],
      bias: Tensor[D] | Option[Tensor[D]] = None,
      stride: Int | (Int, Int) = 1,
      padding: Int | (Int, Int) = 0,
      output_padding: Int | (Int, Int) = 0,
      groups: Int = 1,
      dilation: Int | (Int, Int) = 1
  ): Tensor[D] =
    convTranspose2d(input, weight, bias, stride, padding, output_padding, groups, dilation)

  def convTranspose2d[D <: FloatNN | ComplexNN](
      input: Tensor[D],
      weight: Tensor[D],
      bias: Tensor[D] | Option[Tensor[D]] = None,
      stride: Int | (Int, Int) = 1,
      padding: Int | (Int, Int) = 0,
      outputPadding: Int | (Int, Int) = 0,
      groups: Int = 1,
      dilation: Int | (Int, Int) = 1
  ): Tensor[D] =
    fromNative(
      torchNative.conv_transpose2d(
        input.native,
        weight.native,
        toOptional(bias),
        toArray(stride),
        toArray(padding),
        toArray(outputPadding),
        groups,
        toArray(dilation): _*
      )
    )

  /** Applies a 3D transposed convolution operator over an input image composed of several input
    * planes, sometimes also called “deconvolution”.
    *
    * @group nn_conv
    */
  def conv_transpose3d[D <: FloatNN | ComplexNN](
      input: Tensor[D],
      weight: Tensor[D],
      bias: Tensor[D] | Option[Tensor[D]] = None,
      stride: Int | (Int, Int, Int) = 1,
      padding: Int | (Int, Int, Int) = 0,
      output_padding: Int | (Int, Int, Int) = 0,
      groups: Int = 1,
      dilation: Int | (Int, Int) = 1
  ): Tensor[D] =
    convTranspose3d(input, weight, bias, stride, padding, output_padding, groups, dilation)

  def convTranspose3d[D <: FloatNN | ComplexNN](
      input: Tensor[D],
      weight: Tensor[D],
      bias: Tensor[D] | Option[Tensor[D]] = None,
      stride: Int | (Int, Int, Int) = 1,
      padding: Int | (Int, Int, Int) = 0,
      outputPadding: Int | (Int, Int, Int) = 0,
      groups: Int = 1,
      dilation: Int | (Int, Int) = 1
  ): Tensor[D] =
    fromNative(
      torchNative.conv_transpose3d(
        input.native,
        weight.native,
        toOptional(bias),
        toArray(stride),
        toArray(padding),
        toArray(outputPadding),
        groups,
        toArray(dilation): _*
      )
    )

//  def unfold[D <: FloatNN | ComplexNN](
//      input: Tensor[D],
//      kernel_size: Int | (Int, Int) | (Int, Int, Int),
////      output_size: Int | (Int, Int) | (Int, Int, Int),
//      dilation: Int | (Int, Int) | (Int, Int, Int),
//      padding: Int | (Int, Int) | (Int, Int, Int),
//      stride: Int | (Int, Int) | (Int, Int, Int)
//  ): Tensor[D] =
//    val options = new UnfoldOptions(toNative(kernel_size))
////    options.kernel_size.put(toArray(kernelSize): _*)
//    options.dilation.put(toArray(dilation): _*)
//    options.padding.put(toArray(padding): _*)
//    options.stride.put(toArray(stride): _*)
//    fromNative(
//      torchNative.unfold(
//        input.native,
//        options
//      )
//    )
//
//  def fold[D <: FloatNN | ComplexNN](
//      input: Tensor[D],
//      output_size: Int | (Int, Int) | (Int, Int, Int),
//      kernel_size: Int | (Int, Int) | (Int, Int, Int),
//      dilation: Int | (Int, Int) | (Int, Int, Int),
//      padding: Int | (Int, Int) | (Int, Int, Int),
//      stride: Int | (Int, Int) | (Int, Int, Int)
//  ): Tensor[D] =
//    val options = new FoldOptions(toNative(output_size), toNative(kernel_size))
//    //    options.kernel_size.put(toArray(kernelSize):_*)
//    options.dilation.put(toArray(dilation): _*)
//    options.padding.put(toArray(padding): _*)
//    options.stride.put(toArray(stride): _*)
//    //    options.output_size.put(toArray(outputSize):_*)
//    fromNative(
//      torchNative.fold(
//        input.native,
//        options
//      )
//    )

}
