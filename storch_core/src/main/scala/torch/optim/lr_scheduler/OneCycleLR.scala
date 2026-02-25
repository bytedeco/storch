package torch
package optim
package lr_scheduler

import scala.math.{Pi, cos}
//import torch.optim.Optimizer

/** 1cycle学习率策略 optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None,
  * pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85,
  * max_momentum=0.95, div_factor=25., final_div_factor=1e4, three_phase=False, last_epoch=-1,
  * verbose="deprecated"
  */
class OneCycleLR(
    override val optimizer: Optimizer,
    max_lr: Either[Float, Seq[Float]],
    total_step: Option[Int] = None,
    epochs: Option[Int] = None,
    steps_per_epoch: Option[Int] = None,
    pct_start: Float = 0.3f,
    anneal_strategy: String = "cos",
    cycle_momentum: Boolean = true,
    base_momentum: Either[Float, Seq[Float]] = Left(0.85f),
    max_momentum: Either[Float, Seq[Float]] = Left(0.95f),
    div_factor: Float = 25.0f,
    final_div_factor: Float = 1e4f,
    three_phase: Boolean = false,
    last_epoch: Int = -1,
    use_beta1: Boolean = false,
    verbose: Boolean = false
) extends LRScheduler(optimizer) {
//  override var verbose: Boolean = verbose

  // 验证参数
  val total_steps: Int = total_step match {
    case Some(ts) =>
      require(ts > 0, s"Expected positive integer total_steps, but got $ts")
      ts
    case None =>
      require(
        epochs.isDefined && steps_per_epoch.isDefined,
        "You must define either total_steps OR (epochs AND steps_per_epoch)"
      )
      require(epochs.get > 0, s"Expected positive integer epochs, but got ${epochs.get}")
      require(
        steps_per_epoch.get > 0,
        s"Expected positive integer steps_per_epoch, but got ${steps_per_epoch.get}"
      )
      epochs.get * steps_per_epoch.get
  }

  require(
    pct_start >= 0 && pct_start <= 1,
    s"Expected float between 0 and 1 pct_start, but got $pct_start"
  )
  require(
    anneal_strategy == "cos" || anneal_strategy == "linear",
    s"anneal_strategy must by one of 'cos' or 'linear', instead got $anneal_strategy"
  )

  // 定义退火函数
  val anneal_func: (Float, Float, Float) => Float = anneal_strategy match {
    case "cos"    => _annealing_cos
    case "linear" => _annealing_linear
  }

  // 初始化阶段配置
  val _schedule_phases: Seq[PhaseConfig] = if (three_phase) {
    Seq(
      PhaseConfig(
        end_step = (pct_start * total_steps).toFloat - 1,
        start_lr = "initial_lr",
        end_lr = "max_lr",
        start_momentum = "max_momentum",
        end_momentum = "base_momentum"
      ),
      PhaseConfig(
        end_step = (2 * pct_start * total_steps).toFloat - 2,
        start_lr = "max_lr",
        end_lr = "initial_lr",
        start_momentum = "base_momentum",
        end_momentum = "max_momentum"
      ),
      PhaseConfig(
        end_step = total_steps.toFloat - 1,
        start_lr = "initial_lr",
        end_lr = "min_lr",
        start_momentum = "max_momentum",
        end_momentum = "max_momentum"
      )
    )
  } else {
    Seq(
      PhaseConfig(
        end_step = (pct_start * total_steps).toFloat - 1,
        start_lr = "initial_lr",
        end_lr = "max_lr",
        start_momentum = "max_momentum",
        end_momentum = "base_momentum"
      ),
      PhaseConfig(
        end_step = total_steps.toFloat - 1,
        start_lr = "max_lr",
        end_lr = "min_lr",
        start_momentum = "base_momentum",
        end_momentum = "max_momentum"
      )
    )
  }

  // 初始化学习率变量
  val max_lrs: Seq[Float] = _format_param("max_lr", optimizer, max_lr)
  if (last_epoch == -1) {
    for ((max_lr, group) <- max_lrs.zip(optimizer.param_groups)) {
      val initial_lr = max_lr / div_factor
      group.paramGroupDict("initial_lr") = initial_lr
      group.paramGroupDict("max_lr") = max_lr
      group.paramGroupDict("min_lr") = initial_lr / final_div_factor
    }
  }

  // 初始化动量变量
//  val cycle_momentum: Boolean = cycle_momentum
//  val use_beta1: Boolean = optimizer.defaults.contains("betas")
  var base_momentums: Seq[Float] = Seq.empty
  var max_momentums: Seq[Float] = Seq.empty

  if (cycle_momentum) {
    base_momentums = _format_param("base_momentum", optimizer, base_momentum)
    max_momentums = _format_param("max_momentum", optimizer, max_momentum)

    if (last_epoch == -1) {
      for (
        (m_momentum, b_momentum, group) <- (
          max_momentums,
          base_momentums,
          optimizer.param_groups
        ).zipped
      ) {
//        val m_momentum = m_momentum_b_momentum_group._1
//        val b_momentum = m_momentum_b_momentum_group._2
//        val group = m_momentum_b_momentum_group._3
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

  /** 余弦退火函数
    */
  private def _annealing_cos(start: Float, end: Float, pct: Float): Float = {
    val cos_out = cos(Pi * pct) + 1.0f
    val cosOut = end * 1.0f + (start - end) / 2.0f * cos_out
    cosOut.toFloat
  }

  /** 线性退火函数
    */
  private def _annealing_linear(start: Float, end: Float, pct: Float): Float = {
    (end - start) * pct + start
  }

  override def get_lr(): Seq[Float] = {
    if (!_get_lr_called_within_step) {
      println(
        "Warning: To get the last learning rate computed by the scheduler, please use `get_last_lr()`. "
      )
    }

    val step_num = last_epoch
    if (step_num > total_steps) {
      throw new IllegalArgumentException(
        s"Tried to step $step_num times. The specified number of total steps is $total_steps"
      )
    }

    val lrs = optimizer.param_groups.map { group =>
      var start_step = 0.0f
      var computed_lr = 0.0f
      var computed_momentum = 0.0f
      var found = false

      for (phase <- _schedule_phases) {
        if ((step_num <= phase.end_step || phase == _schedule_phases.last) && !found) {
          val pct = (step_num - start_step) / (phase.end_step - start_step)
          computed_lr = anneal_func(
            group.paramGroupDict(phase.start_lr).asInstanceOf[Float],
            group.paramGroupDict(phase.end_lr).asInstanceOf[Float],
            pct
          )

          if (cycle_momentum) {
            computed_momentum = anneal_func(
              group.paramGroupDict(phase.start_momentum).asInstanceOf[Float],
              group.paramGroupDict(phase.end_momentum).asInstanceOf[Float],
              pct
            )
          }
          found = true
        } else {
          start_step = phase.end_step
        }
      }

      if (cycle_momentum) {
        if (use_beta1) {
          val old_betas = group.paramGroupDict("betas").asInstanceOf[(Float, Float)]
          group.paramGroupDict("betas") = (computed_momentum, old_betas._2)
        } else {
          group.paramGroupDict("momentum") = computed_momentum
        }
      }

      computed_lr
    }

    lrs
  }

  /** 阶段配置内部类
    */
  case class PhaseConfig(
      end_step: Float,
      start_lr: String,
      end_lr: String,
      start_momentum: String,
      end_momentum: String
  )
}
