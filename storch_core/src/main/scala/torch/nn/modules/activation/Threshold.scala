//package torch.nn.modules.activation //ThresholdImpl
package torch
package nn
package modules
package activation

import torch.nn
import torch.nn.modules
import org.bytedeco.pytorch
import torch.internal.NativeConverters.fromNative
import org.bytedeco.pytorch.{ThresholdImpl, ThresholdOptions}

import torch.nn
import torch.nn.modules

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
final class Threshold[D <: DType: Default](
    val threshold: Float,
    val value: Float,
    val inplace: Boolean = false
) extends TensorModule[D]:

  val options = ThresholdOptions(threshold.toDouble, value.toDouble)
  options.inplace().put(inplace)
  options.threshold().put(threshold.toDouble)
  options.value().put(value.toDouble)

  override protected[torch] val nativeModule: ThresholdImpl = new ThresholdImpl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))

  def forward(input: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(input.native))

  override def toString =
    getClass().getSimpleName() + s"(threshold=$threshold, value=$value, inplace=$inplace)"

object Threshold:
  def apply[D <: DType: Default](
      threshold: Float,
      value: Float,
      inplace: Boolean = false
  ): Threshold[D] =
    new Threshold(threshold, value, inplace)
