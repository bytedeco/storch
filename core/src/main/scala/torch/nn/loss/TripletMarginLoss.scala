//package torch.nn.loss
package torch
package nn
package loss

import torch.nn.modules.Module
import torch.internal.NativeConverters.fromNative
import org.bytedeco.pytorch.{
  TripletMarginLossImpl,
  TripletMarginLossOptions,
  LossReduction,
  kMean,
  kSum,
  kNone
}

//class torch.nn.dTripletMarginLoss(margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean')[source]
final class TripletMarginLoss(
    val margin: Double = 1.0,
    val p: Double = 2.0,
    val eps: Double = 1e-06,
    val swap: Boolean = false,
    val reduction: String = "mean",
    val size_average: Option[Boolean] = None,
    val reduce: Option[Boolean] = None
) extends LossFunc {
  private[torch] val options: TripletMarginLossOptions = new TripletMarginLossOptions()

  options.margin().put(margin)
  options.p().put(p)
  options.eps().put(eps)
  options.swap().put(swap)
  val lossReduction = reduction match {
    case "mean" | "Mean" | "MEAN" => new LossReduction(new kMean())
    case "sum" | "Sum" | "SUM"    => new LossReduction(new kSum())
    case "none" | "None" | "NONE" => new LossReduction(new kNone())
    case _ => throw new IllegalArgumentException(s"Unknown reduction $reduction")
  }
  options.reduction().put(lossReduction)

  override private[torch] val nativeModule: TripletMarginLossImpl = TripletMarginLossImpl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  def apply[D <: DType](anchor: Tensor[D], positive: Tensor[D], negative: Tensor[?]): Tensor[D] =
    fromNative(
      nativeModule.forward(anchor.native, positive.native, negative.native)
    )
  def forward[D <: DType](anchor: Tensor[D], positive: Tensor[D], negative: Tensor[?]): Tensor[D] =
    apply(anchor, positive, negative)

  override def apply[D <: DType](inputs: Tensor[D]*)(negative: Tensor[?]): Tensor[D] = {
    val input = inputs.toSeq.head
    val positive = inputs.toSeq.last

    fromNative(
      nativeModule.forward(input.native, positive.native, negative.native) // .output()
    )
  }
  def forward[D <: DType](inputs: Tensor[D]*)(negative: Tensor[?]): Tensor[D] = {
    val input = inputs.toSeq.head
    val positive = inputs.toSeq.last

    fromNative(
      nativeModule.forward(input.native, positive.native, negative.native) // .output()
    )
  }

}

object TripletMarginLoss {

  def apply(
      margin: Double = 1.0,
      p: Double = 2.0,
      eps: Double = 1e-06,
      swap: Boolean = false,
      reduction: String = "mean",
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): TripletMarginLoss =
    new TripletMarginLoss(margin, p, eps, swap, reduction, size_average, reduce)
}
