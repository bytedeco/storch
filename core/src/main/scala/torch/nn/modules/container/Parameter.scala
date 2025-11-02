package torch
package nn
package modules
package container
import org.bytedeco.pytorch
import sourcecode.Name
import scala.collection.mutable.ListBuffer
import org.bytedeco.pytorch.Module
import torch.internal.NativeConverters.fromNative

object Parameter {

  def apply[D <: DType](
      name: String,
      weight: Tensor[D],
      requires_grad: Boolean = true
  )(implicit nativeModule: Module = pytorch.Module()): Tensor[D] =
    nativeModule.register_parameter(name, weight.native, requires_grad)
    weight

  def register_parameter_r[D <: DType](
      weight: Tensor[D],
      requires_grad: Boolean = true,
      n: String = ""
  )(implicit nativeModule: Module = pytorch.Module(), name: sourcecode.Name): Tensor[D] =
    val name_ = if n.trim().isEmpty() then name.value else n.trim()
    nativeModule.register_parameter(name_, weight.native, requires_grad)
    weight

  def register_parameter[D <: DType](
      name: String,
      weight: Tensor[D],
      requires_grad: Boolean = true
  )(implicit nativeModule: Module = pytorch.Module()): Tensor[D] =
    nativeModule.register_parameter(name, weight.native, requires_grad)
    weight
}
