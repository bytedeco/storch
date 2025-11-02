//package torch.nn.loss
package torch
package nn
package loss

import org.bytedeco.pytorch.{
  PoissonNLLLossImpl,
  PoissonNLLLossOptions,
  LossReduction,
  kMean,
  kSum,
  kNone
}
import torch.internal.NativeConverters.fromNative
import torch.nn.modules.Module
/*
 * Parameters
log_input (bool, optional) – if True the loss is computed as
exp

input−target∗log(input+eps)
full (bool, optional) –
whether to compute full loss, i. e. to add the Stirling approximation term
target∗log(target)−target+0.5∗log(2πtarget).
size_average (bool, optional) – Deprecated (see reduction). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is False. Default: True
eps (float, optional) – Small value to avoid evaluation of
reduce  (bool, optional) – Deprecated (see reduction). By default, the losses are averaged or summed over observations for each minibatch depending on size_average. When reduce is False, returns a loss per batch element instead and ignores size_average. Default: True

reduction (str, optional) – Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: 'mean'
 * */
//class torch.nn.PoissonNLLLoss(log_input=True, full=False, size_average=None, eps=1e-08, reduce=None, reduction='mean')[source]
final class PoissonNLLLoss(
    val log_input: Boolean = true,
    val full: Boolean = false,
    val eps: Double = 1e-08,
    val reduction: String = "mean",
    val size_average: Option[Boolean] = None,
    val reduce: Option[Boolean] = None
) extends LossFunc {

  private[torch] val options: PoissonNLLLossOptions = new PoissonNLLLossOptions()
  val lossReduction = reduction match {
    case "mean" | "Mean" | "MEAN" => new LossReduction(new kMean())
    case "sum" | "Sum" | "SUM"    => new LossReduction(new kSum())
    case "none" | "None" | "NONE" => new LossReduction(new kNone())
    case _ => throw new IllegalArgumentException(s"Unknown reduction $reduction")
  }
  options.reduction().put(lossReduction)
  options.log_input().put(log_input)
  options.full().put(full)
  options.eps().put(eps)

  override private[torch] val nativeModule: PoissonNLLLossImpl = PoissonNLLLossImpl(options)

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

object PoissonNLLLoss {

  def apply(
      log_input: Boolean = true,
      full: Boolean = false,
      eps: Double = 1e-08,
      reduction: String = "mean",
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): PoissonNLLLoss = new PoissonNLLLoss(log_input, full, eps, reduction, size_average, reduce)
}
