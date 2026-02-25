package torch

package optim

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{AdagradOptions, OptimizerParamState, AdagradParamState, TensorVector}
import scala.collection.immutable.Iterable

// format: off

/** Implements the AdamW algorithm.
 *CLASStorch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, 
 * initial_accumulator_value=0, eps=1e-10,
 * foreach=None, *, maximize=False, differentiable=False, fused=None)
 */
// format: on
final class Adagrad(
    params: Iterable[Tensor[?]],
    lr: Double = 1e-3,
    lrDecay: Double = 0.999,
    eps: Double = 1e-8,
    weightDecay: Double = 0,
    initialAccumulatorValue: Double = 0.001
) extends Optimizer {
  private val nativeParams: TensorVector = TensorVector(params.map(_.native).toArray*)
  private val options: AdagradOptions = AdagradOptions(lr)

  options.lr().put(lr)
  options.lr_decay().put(lrDecay)
  options.eps().put(eps)
  options.weight_decay().put(weightDecay)
  options.initial_accumulator_value().put(initialAccumulatorValue)

  override val optimizerParamState: OptimizerParamState = new AdagradParamState()
  override private[torch] val native: pytorch.Adagrad = pytorch.Adagrad(nativeParams, options)
}

object Adagrad:
  def apply(
      params: Iterable[Tensor[?]],
      lr: Double = 1e-3,
      lr_decay: Double = 0.999,
      eps: Double = 1e-8,
      weight_decay: Double = 0,
      initial_accumulator_value: Double = 0.001
  ): Adagrad = new Adagrad(params, lr, lr_decay, eps, weight_decay, initial_accumulator_value)
