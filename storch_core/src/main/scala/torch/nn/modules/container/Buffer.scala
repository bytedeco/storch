package torch
package nn
package modules.container
import org.bytedeco.pytorch
import org.bytedeco.pytorch.Module
import torch.internal.NativeConverters.fromNative

object Buffer {

  def apply[D <: DType](name: String, tensor: Tensor[D], persistent: Boolean = false)(implicit
      nativeModule: Module = pytorch.Module()
  ): Tensor[D] = fromNative(nativeModule.register_buffer(name, tensor.native))

  def register_buffer[D <: DType](name: String, tensor: Tensor[D], persistent: Boolean = false)(
      implicit nativeModule: Module = pytorch.Module()
  ): Tensor[D] =
    fromNative(nativeModule.register_buffer(name, tensor.native))
}
