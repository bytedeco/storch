//package torch.nn.loss //BCEWithLogitsLossImpl
package torch
package nn
package loss

import org.bytedeco.pytorch.{
  BCEWithLogitsLossImpl,
  BCEWithLogitsLossOptions,
  LossReduction,
  kMean,
  kSum,
  kNone
}
import torch.internal.NativeConverters.fromNative
import torch.nn.modules.Module

//class torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)[source]
final class BCEWithLogitsLoss(
    val weight: Option[Tensor[?]] = None,
    val reduction: String = "mean",
    val pos_weight: Option[Tensor[?]] = None,
    val size_average: Option[Boolean] = None,
    val reduce: Option[Boolean] = None
) extends LossFunc {

  private[torch] val options: BCEWithLogitsLossOptions = new BCEWithLogitsLossOptions()
  val lossReduction = reduction match {
    case "mean" | "Mean" | "MEAN" => new LossReduction(new kMean())
    case "sum" | "Sum" | "SUM"    => new LossReduction(new kSum())
    case "none" | "None" | "NONE" => new LossReduction(new kNone())
    case _ => throw new IllegalArgumentException(s"Unknown reduction $reduction")
  }
  options.reduction().put(lossReduction)
  if weight.isDefined then options.weight().put(weight.get.native)
  if pos_weight.isDefined then options.pos_weight().put(pos_weight.get.native)
  override private[torch] val nativeModule: BCEWithLogitsLossImpl = BCEWithLogitsLossImpl(options)

  override def hasBias(): Boolean = false
  def weight[D <: DType](): Tensor[D] = fromNative(nativeModule.weight())
  def pos_weight[D <: DType](): Tensor[D] = fromNative(nativeModule.pos_weight())

//  def weight(weight: Tensor[D]) = nativeModule.weight(weight.native)

  def reset(): Unit = nativeModule.reset()

  def forward[D <: DType](input: Tensor[D], target: Tensor[?]): Tensor[D] = apply(input, target)

  def apply[D <: DType](input: Tensor[D], target: Tensor[?]): Tensor[D] = fromNative(
    nativeModule.forward(input.native, target.native)
  )

  override def apply[D <: DType](inputs: Tensor[D]*)(target: Tensor[?]): Tensor[D] = {
    val input = inputs.toSeq.head
    fromNative(
      nativeModule.forward(input.native, target.native)
    )
  }

  def forward[D <: DType](inputs: Tensor[D]*)(target: Tensor[?]): Tensor[D] = {
    val input = inputs.toSeq.head
    fromNative(
      nativeModule.forward(input.native, target.native)
    )
  }

}
object BCEWithLogitsLoss {
  def apply(
      weight: Option[Tensor[?]] = None,
      reduction: String = "mean",
      pos_weight: Option[Tensor[?]] = None,
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): BCEWithLogitsLoss = new BCEWithLogitsLoss(weight, reduction, pos_weight, size_average, reduce)
}
