//package torch.nn.loss
package torch
package nn
package loss

import org.bytedeco.pytorch.{
  KLDivLossImpl,
  KLDivLossOptions,
  LossReduction,
  kMean,
  kSum,
  kBatchMean,
  kNone
}
import torch.internal.NativeConverters.fromNative
import torch.nn.modules.Module

//class torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=False)
final class KLDivLoss(
    val reduction: String = "mean",
    val log_target: Boolean = false,
    val size_average: Option[Boolean] = None,
    val reduce: Option[Boolean] = None
) extends LossFunc {

  private[torch] val options: KLDivLossOptions = reduction match {
    case "mean" | "Mean" | "MEAN"                => new KLDivLossOptions(new kMean())
    case "sum" | "Sum" | "SUM"                   => new KLDivLossOptions(new kSum())
    case "none" | "None" | "NONE"                => new KLDivLossOptions(new kNone)
    case "batchmean" | "BatchMean" | "BATCHMEAN" => new KLDivLossOptions(new kBatchMean())
  }
  val lossReduction = reduction match {
    case "mean" | "Mean" | "MEAN" => new LossReduction(new kMean())
    case "sum" | "Sum" | "SUM"    => new LossReduction(new kSum())
    case "none" | "None" | "NONE" => new LossReduction(new kNone())
//    case "batchmean" | "BatchMean" | "BATCHMEAN" => new LossReduction(new kBatchMean())
    case _ => throw new IllegalArgumentException(s"Unknown reduction $reduction")
  }
  options.reduction().put(lossReduction)
  options.log_target().put(log_target)

  override private[torch] val nativeModule: KLDivLossImpl = KLDivLossImpl(options)

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
object KLDivLoss {
  def apply(
      reduction: String = "mean",
      log_target: Boolean = false,
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): KLDivLoss = new KLDivLoss(reduction, log_target, size_average, reduce)
}
