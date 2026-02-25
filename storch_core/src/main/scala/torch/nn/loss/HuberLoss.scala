//package torch.nn.loss
package torch
package nn
package loss

import org.bytedeco.pytorch.{HuberLossImpl, HuberLossOptions, LossReduction, kMean, kSum, kNone}
import torch.internal.NativeConverters.fromNative
import torch.nn.modules.Module

//class torch.nn.HuberLoss(reduction='mean', delta=1.0)[source]
final class HuberLoss(val reduction: String = "mean", val delta: Double = 1.0) extends LossFunc {

  private[torch] val options: HuberLossOptions = new HuberLossOptions()
  val lossReduction = reduction match {
    case "mean" | "Mean" | "MEAN" => new LossReduction(new kMean())
    case "sum" | "Sum" | "SUM"    => new LossReduction(new kSum())
    case "none" | "None" | "NONE" => new LossReduction(new kNone())
    case _ => throw new IllegalArgumentException(s"Unknown reduction $reduction")
  }
  options.reduction().put(lossReduction)
  options.delta().put(delta)
  override private[torch] val nativeModule: HuberLossImpl = HuberLossImpl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  def forward[D <: DType](input: Tensor[D], target: Tensor[?]): Tensor[D] = apply(input, target)

  def apply[D <: DType](input: Tensor[D], target: Tensor[?]): Tensor[D] = fromNative(
    nativeModule.forward(input.native, target.native)
  )

  override def apply[D <: DType](inputs: Tensor[D]*)(target: Tensor[?]): Tensor[D] = {
    val input = inputs.toSeq.head
    fromNative(
      nativeModule.forward(input.native, target.native) // .output()
    )
  }
  def forward[D <: DType](inputs: Tensor[D]*)(target: Tensor[?]): Tensor[D] = {
    val input = inputs.toSeq.head
    fromNative(
      nativeModule.forward(input.native, target.native) // .output()
    )
  }

}

object HuberLoss {
  def apply(reduction: String = "mean", delta: Double = 1.0): HuberLoss =
    new HuberLoss(reduction, delta)
}
