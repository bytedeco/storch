package torch
package optim
package lr_scheduler

import scala.math.floor
//import torch.optim.Optimizer
//import torch.optim.Optimizer
/** 循环学习率策略 optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, mode='triangular',
  * gamma=1., scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8,
  * max_momentum=0.9, last_epoch=-1, verbose="deprecated"
  */
class CyclicLR(
    override val optimizer: Optimizer,
    base_lr: Either[Float, Seq[Float]],
    max_lr: Either[Float, Seq[Float]],
    step_size_up: Int = 2000,
    step_size_down_option: Option[Int] = None,
    mode: String = "triangular",
    gamma: Float = 1.0f,
    scale_fn: Option[Float => Float] = None,
    scale_mode: String = "cycle",
    cycle_momentum: Boolean = true,
    base_momentum: Either[Float, Seq[Float]] = Left(0.8f),
    max_momentum: Either[Float, Seq[Float]] = Left(0.9f),
    last_epoch: Int = -1,
    use_beta1: Boolean = false,
    verbose: Boolean = false
) extends LRScheduler(optimizer) {
//  override var verbose: Boolean = verbose

  // 处理参数格式
  val _base_lrs: Seq[Float] = _format_param("base_lr", optimizer, base_lr)
  this.base_lrs = _base_lrs
  val max_lrs: Seq[Float] = _format_param("max_lr", optimizer, max_lr)

//  val step_size_up: Float = step_size_up.toFloat
  val step_size_down: Float = step_size_down_option.map(_.toFloat).getOrElse(step_size_up.toFloat)
  val total_size: Float = step_size_up + step_size_down
  val step_ratio: Float = step_size_up / total_size

//  val mode: String = mode
//  val gamma: Float = gamma
  val _scale_fn_custom: Option[Float => Float] = scale_fn
//  val scale_mode: String = scale_mode

//  val cycle_momentum: Boolean = cycle_momentum
//  val use_beta1: Boolean = optimizer.defaults.contains("betas")
  val base_momentums: Seq[Float] =
    if (cycle_momentum) _format_param("base_momentum", optimizer, base_momentum) else Seq.empty
  val max_momentums: Seq[Float] =
    if (cycle_momentum) _format_param("max_momentum", optimizer, max_momentum) else Seq.empty

  // 初始化
  if (last_epoch == -1) {
    // 设置初始学习率
    for ((lr, group) <- base_lrs.zip(optimizer.param_groups)) {
      group.paramGroupDict("lr") = lr
      group.paramGroup.options().set_lr(lr)
    }

    // 设置初始动量
    if (cycle_momentum) {
      for (
        (m_momentum, b_momentum, group) <- (
          max_momentums,
          base_momentums,
          optimizer.param_groups
        ).zipped
      ) {
//      for ((m_momentum, b_momentum, group) <- max_momentums.zip(base_momentums).zip(optimizer.param_groups.map(_.paramGroupDict))) {
        if (use_beta1) {
          val old_betas = group.paramGroupDict("betas").asInstanceOf[(Float, Float)]
          group.paramGroupDict("betas") = (m_momentum, old_betas._2)
        } else {
          group.paramGroupDict("momentum") = m_momentum
        }
        group.paramGroupDict("max_momentum") = m_momentum
        group.paramGroupDict("base_momentum") = b_momentum
      }
    }
  }

  this.last_epoch = last_epoch

  _initial_step()

  /** 格式化参数
    */
  private def _format_param(
      name: String,
      optimizer: Optimizer,
      param: Either[Float, Seq[Float]]
  ): Seq[Float] = {
    param match {
      case Left(value) => Seq.fill(optimizer.param_groups.length)(value)
      case Right(seq) =>
        require(
          seq.length == optimizer.param_groups.length,
          s"expected ${optimizer.param_groups.length} values for $name, got ${seq.length}"
        )
        seq
    }
  }

  /** 获取缩放函数
    */
  def get_scale_fn(x: Float): Float = {
    _scale_fn_custom match {
      case Some(fn) => fn(x)
      case None =>
        mode match {
          case "triangular"  => 1.0f
          case "triangular2" => 1.0f / math.pow(2.0, x - 1).toFloat
          case "exp_range"   => math.pow(gamma, x).toFloat
          case _             => throw new IllegalArgumentException(s"Invalid mode: $mode")
        }
    }
  }

  override def get_lr(): Seq[Float] = {
    if (!_get_lr_called_within_step) {
      println(
        "Warning: To get the last learning rate computed by the scheduler, please use `get_last_lr()`. "
      )
    }

    val cycle = floor(1 + last_epoch / total_size).toFloat
    val x = 1.0f + last_epoch / total_size - cycle
    val scale_factor = if (x <= step_ratio) x / step_ratio else (x - 1.0f) / (step_ratio - 1.0f)

    val lrs = base_lrs.zip(max_lrs).map { case (base_lr, max_lr) =>
      val base_height = (max_lr - base_lr) * scale_factor
      if (scale_mode == "cycle") {
        base_lr + base_height * get_scale_fn(cycle)
      } else {
        base_lr + base_height * get_scale_fn(last_epoch.toFloat)
      }
    }

    // 调整动量
    if (cycle_momentum) {
      val momentums = base_momentums.zip(max_momentums).map { case (base_momentum, max_momentum) =>
        val base_height = (max_momentum - base_momentum) * scale_factor
        if (scale_mode == "cycle") {
          max_momentum - base_height * get_scale_fn(cycle)
        } else {
          max_momentum - base_height * get_scale_fn(last_epoch.toFloat)
        }
      }

      for ((param_group, momentum) <- optimizer.param_groups.zip(momentums)) {
        if (use_beta1) {
          val old_betas = param_group.paramGroupDict("betas").asInstanceOf[(Float, Float)]
          param_group.paramGroupDict("betas") = (momentum, old_betas._2)
        } else {
          param_group.paramGroupDict("momentum") = momentum
        }
      }
    }

    lrs
  }
}
