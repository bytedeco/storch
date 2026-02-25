package torch
package optim
package lr_scheduler

//import torch.optim.Optimizer

/** 每个epoch将学习率乘以gamma因子
  */
class ExponentialLR(
    override val optimizer: Optimizer,
    gamma: Float,
    last_epoch: Int = -1,
    verbose: Boolean = false
) extends LRScheduler(optimizer)
    with ClosedFormLR {
//  override var verbose: Boolean = verbose
//  val gamma: Float = gamma

  // 初始化
  if (last_epoch == -1) {
    // 设置初始学习率
    for (group <- optimizer.param_groups) {
      if (!group.paramGroupDict.contains("initial_lr")) {
        group.paramGroupDict("initial_lr") = group.paramGroup.options.get_lr() // "lr")
      }
    }
  }

//  optimizer.param_groups.map(param => param.options())
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

    if (last_epoch == 0) {
      optimizer.param_groups.map(param => param.paramGroup.options().get_lr().toFloat)
//      optimizer.param_groups.map(_("lr").asInstanceOf[Float])
    } else {
      optimizer.param_groups.map(param => param.paramGroup.options().get_lr().toFloat * gamma)
//      optimizer.param_groups.map(_("lr").asInstanceOf[Float] * gamma)
    }
  }

  override def get_closed_form_lr(): Seq[Float] = {
    base_lrs.map(base_lr => base_lr * math.pow(gamma, last_epoch).toFloat)
  }
}
