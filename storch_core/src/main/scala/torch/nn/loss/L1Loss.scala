//package torch.nn.loss
package torch
package nn
package loss

import org.bytedeco.pytorch.{L1LossImpl, L1LossOptions, LossReduction, kMean, kSum, kNone}
import torch.internal.NativeConverters.fromNative
import torch.nn.modules.Module

//class torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')[source]
final class L1Loss(
    val reduction: String = "mean",
    val size_average: Option[Boolean] = None,
    val reduce: Option[Boolean] = None
) extends LossFunc {

  private[torch] val options: L1LossOptions = new L1LossOptions()
  val lossReduction = reduction match {
    case "mean" | "Mean" | "MEAN" => new LossReduction(new kMean())
    case "sum" | "Sum" | "SUM"    => new LossReduction(new kSum())
    case "none" | "None" | "NONE" => new LossReduction(new kNone())
    case _ => throw new IllegalArgumentException(s"Unknown reduction $reduction")
  }
  options.reduction().put(lossReduction)
  override private[torch] val nativeModule: L1LossImpl = L1LossImpl(options)

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

object L1Loss {

  def apply(
      reduction: String = "mean",
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): L1Loss = new L1Loss(reduction, size_average, reduce)
}
