package torch
package nn
package functional

import org.bytedeco.pytorch.{UnfoldOptions, FoldOptions}
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.*

private[torch] trait Fold {

  def unfold[D <: FloatNN | ComplexNN](
      input: Tensor[D],
      kernel_size: Int | (Int, Int) | (Int, Int, Int),
      //      output_size: Int | (Int, Int) | (Int, Int, Int),
      dilation: Int | (Int, Int) | (Int, Int, Int),
      padding: Int | (Int, Int) | (Int, Int, Int),
      stride: Int | (Int, Int) | (Int, Int, Int)
  ): Tensor[D] =
    val options = new UnfoldOptions(toNative(kernel_size))
    //    options.kernel_size.put(toArray(kernelSize): _*)
    options.dilation.put(toArray(dilation): _*)
    options.padding.put(toArray(padding): _*)
    options.stride.put(toArray(stride): _*)
    fromNative(
      torchNative.unfold(
        input.native,
        options
      )
    )

  def fold[D <: FloatNN | ComplexNN](
      input: Tensor[D],
      output_size: Int | (Int, Int) | (Int, Int, Int),
      kernel_size: Int | (Int, Int) | (Int, Int, Int),
      dilation: Int | (Int, Int) | (Int, Int, Int),
      padding: Int | (Int, Int) | (Int, Int, Int),
      stride: Int | (Int, Int) | (Int, Int, Int)
  ): Tensor[D] =
    val options = new FoldOptions(toNative(output_size), toNative(kernel_size))
    //    options.kernel_size.put(toArray(kernelSize):_*)
    options.dilation.put(toArray(dilation): _*)
    options.padding.put(toArray(padding): _*)
    options.stride.put(toArray(stride): _*)
    //    options.output_size.put(toArray(outputSize):_*)
    fromNative(
      torchNative.fold(
        input.native,
        options
      )
    )

}
