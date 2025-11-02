//package torch.nn.loss
package torch
package nn
package loss

import org.bytedeco.pytorch.{
  MultiMarginLossImpl,
  MultiMarginLossOptions,
  LossReduction,
  kMean,
  kSum,
  kNone
}
import torch.internal.NativeConverters.fromNative
import torch.nn.modules.Module

//class torch.nn.MultiMarginLoss(p=1, margin=1.0, weight=None, size_average=None, reduce=None, reduction='mean')[source]
final class MultiMarginLoss(
    val p: Int = 1,
    val margin: Double = 1.0,
    val weight: Option[Tensor[?]] = None,
    val reduction: String = "mean",
    val size_average: Option[Boolean] = None,
    val reduce: Option[Boolean] = None
) extends LossFunc {

  private[torch] val options: MultiMarginLossOptions = new MultiMarginLossOptions()
  val lossReduction = reduction match {
    case "mean" | "Mean" | "MEAN" => new LossReduction(new kMean())
    case "sum" | "Sum" | "SUM"    => new LossReduction(new kSum())
    case "none" | "None" | "NONE" => new LossReduction(new kNone())
    case _ => throw new IllegalArgumentException(s"Unknown reduction $reduction")
  }
  options.reduction().put(lossReduction)
  options.p().put(p)
  options.margin().put(margin)
  if weight.isDefined then options.weight().put(weight.get.native)

  override private[torch] val nativeModule: MultiMarginLossImpl = MultiMarginLossImpl(options)

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
object MultiMarginLoss {

  def apply(
      p: Int = 1,
      margin: Double = 1.0,
      weight: Option[Tensor[?]] = None,
      reduction: String = "mean",
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): MultiMarginLoss = new MultiMarginLoss(p, margin, weight, reduction, size_average, reduce)
}
