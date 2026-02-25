//package torch.nn.loss
package torch
package nn
package loss

import org.bytedeco.pytorch.{
  TripletMarginWithDistanceLossImpl,
  LossReduction,
  kMean,
  kSum,
  kNone,
  TripletMarginWithDistanceLossOptions
}
import org.bytedeco.javacpp.Pointer
import torch.nn.modules.Module
import torch.internal.NativeConverters.fromNative

//class torch.nn.TripletMarginWithDistanceLoss(*, distance_function=None, margin=1.0, swap=False, reduction='mean')[source]
/*Parameters
distance_function (Callable, optional) – A nonnegative, real-valued function that quantifies the closeness of two tensors. If not specified, nn.PairwiseDistance will be used. Default: None

margin (float, optional)  – A nonnegative margin representing the minimum difference between the positive and negative distances required for the loss to be 0. Larger margins penalize cases where the negative examples are not distant enough from the anchors, relative to the positives. Default:
1
1
.

swap (bool, optional) – Whether to use the distance swap described in the paper Learning shallow convolutional feature descriptors with triplet losses by V. Balntas, E. Riba et al. If True, and if the positive example is closer to the negative example than the anchor is, swaps the positive example and the anchor in the loss computation. Default: False.

reduction (str, optional) – Specifies the (optional) reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed. Default: 'mean'
 */

final class TripletMarginWithDistanceLoss(
    val distance_function: Option[Pointer] = None,
    val margin: Double = 1.0,
    val swap: Boolean = false,
    val reduction: String = "mean"
) extends LossFunc {

  private[torch] val options: TripletMarginWithDistanceLossOptions =
    new TripletMarginWithDistanceLossOptions()
  options.swap().put(swap)
  options.margin().put(margin)
  val lossReduction = reduction match {
    case "mean" | "Mean" | "MEAN" => new LossReduction(new kMean())
    case "sum" | "Sum" | "SUM"    => new LossReduction(new kSum())
    case "none" | "None" | "NONE" => new LossReduction(new kNone())
    case _ => throw new IllegalArgumentException(s"Unknown reduction $reduction")
  }
  options.reduction().put(lossReduction)
  if (distance_function.isDefined) options.distance_function().put(distance_function.get)

  override private[torch] val nativeModule: TripletMarginWithDistanceLossImpl =
    TripletMarginWithDistanceLossImpl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  def apply[D <: DType](anchor: Tensor[D], positive: Tensor[?], negative: Tensor[?]): Tensor[D] =
    fromNative(
      nativeModule.forward(anchor.native, positive.native, negative.native)
    )
  def forward[D <: DType](anchor: Tensor[D], positive: Tensor[D], negative: Tensor[?]): Tensor[D] =
    apply(anchor, positive, negative)

  override def apply[D <: DType](inputs: Tensor[D]*)(negative: Tensor[?]): Tensor[D] = {
    val anchor = inputs.toSeq.head
    val positive = inputs.toSeq.last
    fromNative(
      nativeModule.forward(anchor.native, positive.native, negative.native) // .output()
    )
  }
  def forward[D <: DType](inputs: Tensor[D]*)(negative: Tensor[?]): Tensor[D] = {
    val anchor = inputs.toSeq.head
    val positive = inputs.toSeq.last
    fromNative(
      nativeModule.forward(anchor.native, positive.native, negative.native) // .output()
    )
  }

}

object TripletMarginWithDistanceLoss {
  def apply(
      distance_function: Option[Pointer] = None,
      margin: Double = 1.0,
      swap: Boolean = false,
      reduction: String = "mean"
  ): TripletMarginWithDistanceLoss =
    new TripletMarginWithDistanceLoss(distance_function, margin, swap, reduction)
}
