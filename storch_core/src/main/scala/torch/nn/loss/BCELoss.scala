//package torch.nn.loss //BCELossImpl
package torch
package nn
package loss

import org.bytedeco.pytorch.{BCELossImpl, BCELossOptions, LossReduction, kMean, kSum, kNone}
import torch.internal.NativeConverters.fromNative
import torch.nn.modules.Module
//Parameters
//weight (Tensor, optional) – a manual rescaling weight given to the loss of each batch element. If given, has to be a Tensor of size nbatch.
//
//size_average (bool, optional) – Deprecated (see reduction). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is False. Default: True
//
//reduce (bool, optional) – Deprecated (see reduction). By default, the losses are averaged or summed over observations for each minibatch depending on size_average. When reduce is False, returns a loss per batch element instead and ignores size_average. Default: True
//
//reduction (str, optional) – Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output,
// 'sum': the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: 'mean'

//class torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
final class BCELoss(
    val weight: Option[Tensor[?]] = None,
    val reduction: String = "mean",
    val size_average: Option[Boolean] = None,
    val reduce: Option[Boolean] = None
) extends LossFunc {

  private[torch] val options: BCELossOptions = new BCELossOptions()
  val lossReduction = reduction match {
    case "mean" | "Mean" | "MEAN" => new LossReduction(new kMean())
    case "sum" | "Sum" | "SUM"    => new LossReduction(new kSum())
    case "none" | "None" | "NONE" => new LossReduction(new kNone())
    case _ => throw new IllegalArgumentException(s"Unknown reduction $reduction")
  }
  options.reduction().put(lossReduction)
  if weight.isDefined then options.weight().put(weight.get.native)

  override private[torch] val nativeModule: BCELossImpl = BCELossImpl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

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

  def forward[D <: DType](input: Tensor[D], target: Tensor[?]): Tensor[D] = apply(input, target)

  def apply[D <: DType](input: Tensor[D], target: Tensor[?]): Tensor[D] = fromNative(
    nativeModule.forward(input.native, target.native)
  )
}
object BCELoss {
  def apply(
      weight: Option[Tensor[?]] = None,
      reduction: String = "mean",
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): BCELoss = new BCELoss(weight, reduction, size_average, reduce)
}
