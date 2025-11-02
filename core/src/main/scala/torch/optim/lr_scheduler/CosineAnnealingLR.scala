package torch
package optim
package lr_scheduler

//import torch.optim.Optimizer
import scala.math.{Pi, cos}

/** 使用余弦退火策略调整学习率 optimizer, T_max, eta_min=0, last_epoch=-1, verbose="deprecated"
  */
class CosineAnnealingLR(
    override val optimizer: Optimizer,
    T_max: Int,
    eta_min: Float = 0.0f,
    last_epoch: Int = -1,
    verbose: Boolean = false
) extends LRScheduler(optimizer)
    with ClosedFormLR {
//  override var verbose: Boolean = verbose
//  val t_max: Int = t_max
//  val eta_min: Float = eta_min

  // 初始化
  if (last_epoch == -1) {
    // 设置初始学习率
    for (group <- optimizer.param_groups) {
      if (!group.paramGroupDict.contains("initial_lr")) {
        group.paramGroupDict("initial_lr") = group.paramGroupDict("lr")
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

    if (last_epoch == 0) {
      optimizer.param_groups.map(param => param.paramGroup.options().get_lr().toFloat)
    } else if (_step_count == 1 && last_epoch > 0) {
      base_lrs.zip(optimizer.param_groups).map { case (base_lr, group) =>
        val lr = eta_min + (base_lr - eta_min) * (1 + cos(last_epoch * Pi / T_max)) / 2.0f
        lr.toFloat
      }
    } else if ((last_epoch - 1 - T_max) % (2 * T_max) == 0) {
      base_lrs.zip(optimizer.param_groups).map { case (base_lr, group) =>
        val lr =
          group.paramGroup.options().get_lr().asInstanceOf[Float] + (base_lr - eta_min) * (1 - cos(
            Pi / T_max
          )) / 2.0f
        lr.toFloat
      }
    } else {
      optimizer.param_groups.map(group =>
        val lr = (1 + cos(Pi * last_epoch / T_max)) / (1 + cos(Pi * (last_epoch - 1) / T_max)) *
          (group.paramGroupDict("lr").asInstanceOf[Float] - eta_min) + eta_min
        lr.toFloat
      )
    }
  }

  override def get_closed_form_lr(): Seq[Float] = {
    base_lrs.map(base_lr => {
      val bash_lr = eta_min + (base_lr - eta_min) * (1 + cos(Pi * last_epoch / T_max)) / 2.0f
      bash_lr.toFloat
    })
  }
}
