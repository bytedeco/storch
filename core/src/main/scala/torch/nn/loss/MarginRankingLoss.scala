//package torch.nn.loss
package torch
package nn
package loss

import org.bytedeco.pytorch.{
  MarginRankingLossImpl,
  MarginRankingLossOptions,
  LossReduction,
  kMean,
  kSum,
  kNone
}
import torch.nn.modules.Module
import torch.internal.NativeConverters.fromNative
import org.bytedeco.pytorch.MarginRankingLossImpl

//class torch.nn.MarginRankingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')[source]

final class MarginRankingLoss(
    val margin: Double = 0.0,
    val reduction: String = "mean",
    val size_average: Option[Boolean] = None,
    val reduce: Option[Boolean] = None
) extends LossFunc {

  private[torch] val options: MarginRankingLossOptions = new MarginRankingLossOptions()
  val lossReduction = reduction match {
    case "mean" | "Mean" | "MEAN" => new LossReduction(new kMean())
    case "sum" | "Sum" | "SUM"    => new LossReduction(new kSum())
    case "none" | "None" | "NONE" => new LossReduction(new kNone())
    case _ => throw new IllegalArgumentException(s"Unknown reduction $reduction")
  }
  options.reduction().put(lossReduction)
  options.margin().put(margin)
  override private[torch] val nativeModule: MarginRankingLossImpl = MarginRankingLossImpl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  def apply[D <: DType](input1: Tensor[D], input2: Tensor[D], target: Tensor[?]): Tensor[D] =
    fromNative(
      nativeModule.forward(input1.native, input2.native, target.native)
    )
  def forward[D <: DType](input1: Tensor[D], input2: Tensor[D], target: Tensor[?]): Tensor[D] =
    apply(input1, input2, target)

  override def apply[D <: DType](inputs: Tensor[D]*)(target: Tensor[?]): Tensor[D] = {
    val input1 = inputs.toSeq.head
    val input2 = inputs.toSeq.last
    fromNative(
      nativeModule.forward(input1.native, input2.native, target.native) // .output()
    )
  }
  def forward[D <: DType](inputs: Tensor[D]*)(target: Tensor[?]): Tensor[D] = {
    val input1 = inputs.toSeq.head
    val input2 = inputs.toSeq.last
    fromNative(
      nativeModule.forward(input1.native, input2.native, target.native) // .output()
    )
  }

}

object MarginRankingLoss {

  def apply(
      margin: Double = 0.0,
      reduction: String = "mean",
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): MarginRankingLoss = new MarginRankingLoss(margin, reduction, size_average, reduce)
}
