package torch
package optim
package lr_scheduler

//import torch.optim.Optimizer

/** 每隔指定的epoch数将学习率乘以gamma因子
  */
class StepLR(
    override val optimizer: Optimizer,
    step_size: Int,
    gamma: Float = 0.1f,
    last_epoch: Int = -1,
    verbose: Boolean = false
) extends LRScheduler(optimizer)
    with ClosedFormLR {
//  override var verbose: Boolean = verbose
//  val step_size: Int = step_size
//  val gamma: Float = gamma

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

    if (last_epoch == 0 || last_epoch % step_size != 0) {
      optimizer.param_groups.map(param => param.paramGroup.options().get_lr().toFloat)
    } else {
      optimizer.param_groups.map(param => param.paramGroup.options().get_lr().toFloat * gamma)
    }
  }

  override def get_closed_form_lr(): Seq[Float] = {
    base_lrs.map(base_lr => base_lr * math.pow(gamma, last_epoch / step_size).toFloat)
  }
}
