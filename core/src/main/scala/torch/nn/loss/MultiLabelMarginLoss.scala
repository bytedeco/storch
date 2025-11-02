//package torch.nn.loss
package torch
package nn
package loss

import org.bytedeco.pytorch.{
  MultiLabelMarginLossImpl,
  MultiLabelMarginLossOptions,
  LossReduction,
  kMean,
  kSum,
  kNone
}
import torch.nn.modules.Module
import torch.internal.NativeConverters.fromNative

//class torch.nn.MultiLabelMarginLoss(size_average=None, reduce=None, reduction='mean')
final class MultiLabelMarginLoss(
    val reduction: String = "mean",
    val size_average: Option[Boolean] = None,
    val reduce: Option[Boolean] = None
) extends LossFunc {

  private[torch] val options: MultiLabelMarginLossOptions = new MultiLabelMarginLossOptions()
  val lossReduction = reduction match {
    case "mean" | "Mean" | "MEAN" => new LossReduction(new kMean())
    case "sum" | "Sum" | "SUM"    => new LossReduction(new kSum())
    case "none" | "None" | "NONE" => new LossReduction(new kNone())
    case _ => throw new IllegalArgumentException(s"Unknown reduction $reduction")
  }
  options.reduction().put(lossReduction)
  override private[torch] val nativeModule: MultiLabelMarginLossImpl = MultiLabelMarginLossImpl(
    options
  )

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  def apply[D <: DType](input: Tensor[D], target: Tensor[?]): Tensor[D] = fromNative(
    nativeModule.forward(input.native, target.native)
  )
  def forward[D <: DType](input: Tensor[D], target: Tensor[?]): Tensor[D] = apply(input, target)

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

object MultiLabelMarginLoss {

  def apply(
      reduction: String = "mean",
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): MultiLabelMarginLoss = new MultiLabelMarginLoss(reduction, size_average, reduce)
}
