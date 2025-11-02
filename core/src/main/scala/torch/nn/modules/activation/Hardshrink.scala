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
import org.bytedeco.pytorch.{HardshrinkImpl, HardshrinkOptions}
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
final class Hardshrink[D <: DType: Default](val lambda: Float = 0.5f) extends TensorModule[D]:

  val options = HardshrinkOptions(lambda.toDouble)
  options.lambda().put(lambda.toDouble)

  override protected[torch] val nativeModule: HardshrinkImpl = new HardshrinkImpl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))

  def forward(input: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(input.native))

  override def toString = getClass().getSimpleName() + s"(lambda=$lambda)"

object Hardshrink:
  def apply[D <: DType: Default](lambda: Float = 0.5f): Hardshrink[D] = new Hardshrink(lambda)
