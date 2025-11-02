//package torch.nn.loss
package torch
package nn
package loss

import org.bytedeco.pytorch.{
  SmoothL1LossImpl,
  SmoothL1LossOptions,
  LossReduction,
  kMean,
  kSum,
  kNone
}
import torch.internal.NativeConverters.fromNative
import torch.nn.modules.Module

//class torchd.nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=1.0)[source]
final class SmoothL1Loss(
    val reduction: String = "mean",
    val beta: Double = 1.0,
    val size_average: Option[Boolean] = None,
    val reduce: Option[Boolean] = None
) extends LossFunc {

  private[torch] val options: SmoothL1LossOptions = new SmoothL1LossOptions()
  val lossReduction = reduction match {
    case "mean" | "Mean" | "MEAN" => new LossReduction(new kMean())
    case "sum" | "Sum" | "SUM"    => new LossReduction(new kSum())
    case "none" | "None" | "NONE" => new LossReduction(new kNone())
    case _ => throw new IllegalArgumentException(s"Unknown reduction $reduction")
  }
  options.reduction().put(lossReduction)
  options.beta().put(beta)
  override private[torch] val nativeModule: SmoothL1LossImpl = SmoothL1LossImpl(options)

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

object SmoothL1Loss {

  def apply(
      reduction: String = "mean",
      beta: Double = 1.0,
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): SmoothL1Loss = new SmoothL1Loss(reduction, beta, size_average, reduce)
}
