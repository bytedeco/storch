package torch.optim.lr_scheduler

import scala.math.{Pi, cos}
//import torch.optim.Optimizer
import torch.optim.Optimizer

/** 带热重启的余弦退火学习率调度器 optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose="deprecated"
  */
class CosineAnnealingWarmRestarts(
    override val optimizer: Optimizer,
    T_0: Int,
    T_mult: Int = 1,
    eta_min: Float = 0.0f,
    last_epoch: Int = -1,
    verbose: Boolean = false
) extends LRScheduler(optimizer) {
//  override var verbose: Boolean = verbose
  require(T_0 > 0 && T_0.isInstanceOf[Int], s"Expected positive integer T_0, but got $T_0")
  require(T_mult >= 1 && T_mult.isInstanceOf[Int], s"Expected integer T_mult >= 1, but got $T_mult")

//  val T_0: Int = T_0
  var t_i: Int = T_0
//  val T_mult: Int = T_mult
//  val eta_min: Float = eta_min
  var t_cur: Int = last_epoch

  // 初始化
  if (last_epoch == -1) {
    // 设置初始学习率
    for (group <- optimizer.param_groups) {
      if (!group.paramGroupDict.contains("initial_lr")) {
        group.paramGroupDict("initial_lr") = group.paramGroup.options().get_lr()
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

    base_lrs.map(base_lr =>
      val lr = eta_min * 1.0f + (base_lr - eta_min) * (1 + cos(Pi * t_cur / t_i)) * 1.0f / 2.0f
      lr.toFloat
    )
  }

  override def step(epoch: Option[Int] = None): Unit = {
    // 处理epoch参数
    val current_epoch = epoch match {
      case Some(e) =>
        if (e < 0) {
          throw new IllegalArgumentException(s"Expected non-negative epoch, but got $e")
        }
        e.toFloat
      case None =>
        if (last_epoch < 0) 0.0f else (last_epoch + 1).toFloat
    }

    // 计算t_cur和t_i
    if (epoch.isDefined) {
      if (current_epoch >= T_0) {
        if (T_mult == 1) {
          t_cur = (current_epoch % T_0).toInt
        } else {
          val n = log((current_epoch / T_0 * (T_mult - 1) + 1), T_mult).toInt
          t_cur = (current_epoch - T_0 * (math.pow(T_mult, n) - 1) / (T_mult - 1)).toInt
          t_i = T_0 * math.pow(T_mult, n).toInt
        }
      } else {
        t_i = T_0
        t_cur = current_epoch.toInt
      }
    } else {
      t_cur += 1
      if (t_cur >= t_i) {
        t_cur -= t_i
        t_i *= T_mult
      }
    }

    last_epoch = current_epoch.floor.toInt

    // 更新学习率
    try {
      _get_lr_called_within_step = true
      val values = get_lr()

      for ((param_group, lr) <- optimizer.param_groups.zip(values)) {
//        param_group("lr") = lr
        param_group.paramGroup.options().set_lr(lr)
        param_group.paramGroupDict("lr") = lr
      }

      _last_lr = optimizer.param_groups.map(param => param.paramGroup.options().get_lr().toFloat)
    } finally {
      _get_lr_called_within_step = false
    }
  }

  /** 计算对数，底数为base
    */
  private def log(value: Double, base: Double): Double = {
    math.log(value) / math.log(base)
  }
}
