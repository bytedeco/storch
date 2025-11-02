//package torch.nn.loss
package torch
package nn
package loss

import org.bytedeco.pytorch.{CTCLossImpl, CTCLossOptions, LossReduction, kMean, kSum, kNone}
import torch.internal.NativeConverters.fromNative
import torch.nn.modules.Module
//class torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)[source]
final class CTCLoss(
    val blank: Long = 0,
    val reduction: String = "mean",
    val zero_infinity: Boolean = false
) extends LossFunc {

  private[torch] val options: CTCLossOptions = new CTCLossOptions()
  val lossReduction = reduction match {
    case "mean" | "Mean" | "MEAN" => new LossReduction(new kMean())
    case "sum" | "Sum" | "SUM"    => new LossReduction(new kSum())
    case "none" | "None" | "NONE" => new LossReduction(new kNone())
    case _ => throw new IllegalArgumentException(s"Unknown reduction $reduction")
  }
  options.reduction().put(lossReduction)
  options.blank().put(blank)
  options.zero_infinity().put(zero_infinity)
  override private[torch] val nativeModule: CTCLossImpl = CTCLossImpl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  def forward[D <: DType](
      input: Tensor[D],
      target: Tensor[?],
      w: Tensor[?],
      k: Tensor[D]
  ): Tensor[D] = apply(input, target, w, k)

  def apply[D <: DType](
      input: Tensor[D],
      target: Tensor[?],
      w: Tensor[?],
      k: Tensor[D]
  ): Tensor[D] = fromNative(
    nativeModule.forward(input.native, target.native, w.native, k.native)
  )

  override def apply[D <: DType](inputs: Tensor[D]*)(target: Tensor[?]): Tensor[D] = {
    val input = inputs.toSeq.head
    val k = inputs.toSeq.last
    val w = inputs.toSeq.dropRight(1).head
    fromNative(
      nativeModule.forward(input.native, target.native, w.native, k.native) // .output()
    )
  }
  def forward[D <: DType](inputs: Tensor[D]*)(target: Tensor[?]): Tensor[D] = {
    val input = inputs.toSeq.head
    val k = inputs.toSeq.last
    val w = inputs.toSeq.dropRight(1).head
    fromNative(
      nativeModule.forward(input.native, target.native, w.native, k.native) // .output()
    )
  }
}

object CTCLoss {
  def apply(blank: Int = 0, reduction: String = "mean", zero_infinity: Boolean = false): CTCLoss =
    new CTCLoss(blank, reduction, zero_infinity)

}
