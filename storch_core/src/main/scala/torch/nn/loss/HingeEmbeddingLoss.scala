//package torch.nn.loss
package torch
package nn
package loss

import torch.nn.modules.Module
import torch.internal.NativeConverters.fromNative
import org.bytedeco.pytorch.{
  HingeEmbeddingLossImpl,
  HingeEmbeddingLossOptions,
  LossReduction,
  kMean,
  kSum,
  kNone
}
//class torch.nn.HingeEmbeddingLoss(margin=1.0, size_average=None, reduce=None, reduction='mean')[source]
final class HingeEmbeddingLoss(
    val margin: Double = 1.0,
    val reduction: String = "mean",
    val size_average: Option[Boolean] = None,
    val reduce: Option[Boolean] = None
) extends LossFunc {

  private[torch] val options: HingeEmbeddingLossOptions = new HingeEmbeddingLossOptions()
  val lossReduction = reduction match {
    case "mean" | "Mean" | "MEAN" => new LossReduction(new kMean())
    case "sum" | "Sum" | "SUM"    => new LossReduction(new kSum())
    case "none" | "None" | "NONE" => new LossReduction(new kNone())
    case _ => throw new IllegalArgumentException(s"Unknown reduction $reduction")
  }
  options.reduction().put(lossReduction)
  options.margin().put(margin)
  override private[torch] val nativeModule: HingeEmbeddingLossImpl = HingeEmbeddingLossImpl(options)

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

object HingeEmbeddingLoss {

  def apply(
      margin: Double = 1.0,
      reduction: String = "mean",
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): HingeEmbeddingLoss = new HingeEmbeddingLoss(margin, reduction, size_average, reduce)
}
