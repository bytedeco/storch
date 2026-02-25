package torch.optim

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{RMSpropOptions, OptimizerParamState, RMSpropParamState, TensorVector}
import torch.Tensor

import scala.collection.immutable.Iterable

// format: off

/** Implements the AdamW algorithm.
 * torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08,
 * weight_decay=0, momentum=0, centered=False,
 * capturable=False, foreach=None, maximize=False, differentiable=False)
 */
// format: on
final class RMSprop(
    params: Iterable[Tensor[?]],
    lr: Double = 1e-3,
    alpha: Double = 0.999,
    eps: Double = 1e-8,
    weightDecay: Double = 0,
    momentum: Double = 0,
    centered: Boolean = false
) extends Optimizer {
  private val nativeParams: TensorVector = TensorVector(params.map(_.native).toArray*)
  private val options: RMSpropOptions = RMSpropOptions(lr)
  options.alpha().put(alpha)
  options.eps().put(eps)
  options.weight_decay().put(weightDecay)
  options.momentum().put(momentum)
  options.centered().put(centered)
  override val optimizerParamState: OptimizerParamState = new RMSpropParamState()
  override private[torch] val native: pytorch.RMSprop = pytorch.RMSprop(nativeParams, options)
}

object RMSprop:
  def apply(
      params: Iterable[Tensor[?]],
      lr: Double = 1e-3,
      alpha: Double = 0.999,
      eps: Double = 1e-8,
      weight_decay: Double = 0,
      momentum: Double = 0,
      centered: Boolean = false
  ): RMSprop = new RMSprop(params, lr, alpha, eps, weight_decay, momentum, centered)
