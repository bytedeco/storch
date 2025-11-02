//package torch.nn.modules.activation //SoftsignImpl
package torch
package nn
package modules
package activation

import org.bytedeco.pytorch
import torch.internal.NativeConverters.fromNative
import org.bytedeco.pytorch.{SoftsignImpl, ELUOptions}

import org.bytedeco.pytorch.TanhImpl
import torch.internal.NativeConverters.fromNative

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
final class Softsign[D <: DType: Default] extends TensorModule[D]:

  override protected[torch] val nativeModule: SoftsignImpl = new SoftsignImpl()

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))

  def forward(input: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(input.native))

  override def toString = getClass().getSimpleName()

object Softsign:
  def apply[D <: DType: Default](): Softsign[D] = new Softsign()
