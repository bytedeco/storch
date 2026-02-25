package torch
package optim
package lr_scheduler

import scala.math.min
import torch.optim.Optimizer

/** 线性调整学习率的乘法因子直到达到指定的迭代次数
  */
class LinearLR(
    override val optimizer: Optimizer,
    start_factor: Float = 1.0f / 3.0f,
    end_factor: Float = 1.0f,
    total_iters: Int = 5,
    last_epoch: Int = -1,
    verbose: Boolean = false
) extends LRScheduler(optimizer)
    with ClosedFormLR {
//  override var verbose: Boolean = verbose
  require(
    start_factor > 0 && start_factor <= 1.0,
    "Starting multiplicative factor expected to be greater than 0 and less or equal to 1."
  )
  require(
    end_factor >= 0 && end_factor <= 1.0,
    "Ending multiplicative factor expected to be between 0 and 1."
  )
//  val start_factor: Float = start_factor
//  val end_factor: Float = end_factor
//  val total_iters: Int = total_iters

  // 初始化
  if (last_epoch == -1) {
    // 设置初始学习率
    for (group <- optimizer.param_groups) {
      if (!group.paramGroupDict.contains("initial_lr")) {
        group.paramGroupDict("initial_lr") = group.paramGroup.options().get_lr() // ("lr")
      }
    }
  }

  base_lrs =
    optimizer.param_groups.map(param => param.paramGroupDict("initial_lr").asInstanceOf[Float])
  this.last_epoch = last_epoch

  _initial_step()

  override def get_lr(): Seq[Float] = {
    if (!_get_lr_called_within_step) {
      println(
        "Warning: To get the last learning rate computed by the scheduler, please use `get_last_lr()`. "
      )
    }

    last_epoch match {
      case 0 =>
        optimizer.param_groups.map(param =>
          param.paramGroup.options().get_lr().toFloat * start_factor
        )
      case _ if last_epoch > total_iters =>
        optimizer.param_groups.map(param => param.paramGroup.options().get_lr().toFloat)
      case _ =>
        val progress = (last_epoch - 1).toFloat / total_iters
        val current_factor = start_factor + (end_factor - start_factor) * progress
        optimizer.param_groups.map(param =>
          param.paramGroup
            .options()
            .get_lr()
            .toFloat / current_factor * (current_factor + (end_factor - start_factor) / total_iters)
        )
    }
  }

  override def get_closed_form_lr(): Seq[Float] = {
    val progress = min(last_epoch, total_iters).toFloat / total_iters
    val current_factor = start_factor + (end_factor - start_factor) * progress
    base_lrs.map(base_lr => base_lr * current_factor)
  }
}
