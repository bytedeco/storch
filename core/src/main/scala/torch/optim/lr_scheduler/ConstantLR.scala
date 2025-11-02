package torch
package optim
package lr_scheduler

//import torch.optim.Optimizer
//import torch.optim.Optimizer
//import torch.{ClosedFormLR, LRScheduler, optimizer}

/** 在指定的迭代次数内将学习率乘以一个小的常数因子
  */
class ConstantLR(
    override val optimizer: Optimizer,
    factor: Float = 1.0f / 3.0f,
    total_iters: Int = 5,
    last_epoch: Int = -1,
    verbose: Boolean = false
) extends LRScheduler(optimizer)
    with ClosedFormLR {
//  override var verbose: Boolean = verbose
  require(
    factor > 0 && factor <= 1.0,
    "Constant multiplicative factor expected to be between 0 and 1."
  )
//  val factor: Float = factor
//  val total_iters: Int = total_iters

  // 初始化
  if (last_epoch == -1) {
    // 设置初始学习率
    for (group <- optimizer.param_groups) {
      if (!group.paramGroupDict.contains("initial_lr")) {
        group.paramGroupDict("initial_lr") = group.paramGroupDict("lr")
      }
    }
  }

//  base_lrs = optimizer.param_groups.map(_("initial_lr").asInstanceOf[Float])
  base_lrs =
    optimizer.param_groups.map(param => param.paramGroupDict("initial_lr").asInstanceOf[Float])
//  optimizer.param_groups.
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
        optimizer.param_groups.map(param => param.paramGroup.options().get_lr().toFloat * factor)
      case _ if last_epoch != total_iters =>
        optimizer.param_groups.map(param => param.paramGroup.options().get_lr().toFloat)
      case _ =>
        optimizer.param_groups.map(param =>
          param.paramGroup.options().get_lr().toFloat * (1.0f / factor)
        )
    }
  }

  override def get_closed_form_lr(): Seq[Float] = {
    val effective_factor = if (last_epoch >= total_iters) 1.0f else factor
    base_lrs.map(base_lr => base_lr * effective_factor)
  }
}
