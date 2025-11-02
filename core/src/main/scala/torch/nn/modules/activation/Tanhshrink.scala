//package torch.nn.modules.activation //TanhshrinkImpl
package torch
package nn
package modules
package activation

//import torch.Tensor.fromNative
import torch.nn
import torch.nn.modules

import org.bytedeco.pytorch
import torch.internal.NativeConverters.fromNative
import org.bytedeco.pytorch.{TanhshrinkImpl, ELUOptions}
import org.bytedeco.pytorch.TanhImpl

/** Applies the Hyperbolic Tangent (Tanh) function element-wise. Tanh is defined as::
  *
  * TODO LaTeX
  *
  * Example:
  *
  * ```scala sc
  * import torch.*
  * val m = nn.Tanh()
  * val input = torch.randn(Seq(2))
  * val output = m(input)
  * ```
  */
final class Tanhshrink[D <: DType: Default] extends TensorModule[D]:

  override protected[torch] val nativeModule: TanhshrinkImpl = new TanhshrinkImpl()

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))

  def forward(input: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(input.native))

  override def toString = getClass().getSimpleName()

object Tanhshrink:
  def apply[D <: DType: Default](): Tanhshrink[D] = new Tanhshrink()
