package torch
package optim
package lr_scheduler

//import torch.optim.Optimizer
import scala.math.min

/** 使用多项式函数衰减学习率
  */
class PolynomialLR(
    override val optimizer: Optimizer,
    total_iters: Int = 5,
    power: Float = 1.0f,
    last_epoch: Int = -1,
    verbose: Boolean = false
) extends LRScheduler(optimizer)
    with ClosedFormLR {
//  override var verbose: Boolean = verbose
//  val total_iters: Int = total_iters
//  val power: Float = power

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

    if (last_epoch == 0 || last_epoch > total_iters) {
      optimizer.param_groups.map(param => param.paramGroup.options().get_lr().toFloat)
    } else {
      val decay_factor = math
        .pow(
          (1.0f - last_epoch.toFloat / total_iters) / (1.0f - (last_epoch - 1).toFloat / total_iters),
          power
        )
        .toFloat
      optimizer.param_groups.map(param =>
        param.paramGroup.options().get_lr().toFloat * decay_factor
      )
    }
  }

  override def get_closed_form_lr(): Seq[Float] = {
    val progress = min(last_epoch, total_iters).toFloat / total_iters
    base_lrs.map(base_lr => base_lr * math.pow(1.0f - progress, power).toFloat)
  }
}
