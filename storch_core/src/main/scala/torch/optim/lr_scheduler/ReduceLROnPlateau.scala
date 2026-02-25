package torch
package optim
package lr_scheduler

//import torch.optim.Optimizer

import scala.math.max

/** 当指标停止改善时降低学习率 optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4,
  * threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8, verbose="deprecated"
  */
class ReduceLROnPlateau(
    override val optimizer: Optimizer,
    mode: String = "min",
    factor: Float = 0.1f,
    patience: Int = 10,
    threshold: Float = 1e-4f,
    threshold_mode: String = "rel",
    cooldown: Int = 0,
    min_lr: Either[Float, Seq[Float]] = Left(0.0f),
    eps: Float = 1e-8f,
    verbose: Boolean = false
) extends LRScheduler(optimizer) {
//  override var verbose: Boolean = verbose
  require(factor < 1.0, "Factor should be < 1.0.")
//
//  val factor: Float = factor
//  val patience: Int = patience
//  val cooldown: Int = cooldown

//  val mode: String = mode
//  val threshold: Float = threshold
//  val threshold_mode: String = threshold_mode
  var best: Float = 0.0f
  var num_bad_epochs: Int = 0
  var mode_worse: Float = 0.0f
  var cooldown_counter: Int = 0
//  val eps: Float = eps
  val min_lrs: Seq[Float] = min_lr match {
    case Left(value) => Seq.fill(optimizer.param_groups.length)(value)
    case Right(seq) =>
      require(
        seq.length == optimizer.param_groups.length,
        s"expected ${optimizer.param_groups.length} min_lrs, got ${seq.length}"
      )
      seq
  }

  // 初始化
  _init_is_better(mode, threshold, threshold_mode)
  _reset()
  _last_lr = optimizer.param_groups.map(param => param.paramGroup.options().get_lr().toFloat)

  /** 重置计数器
    */
  def _reset(): Unit = {
    best = mode_worse
    cooldown_counter = 0
    num_bad_epochs = 0
  }

  /** 执行一步学习率更新
    */
  def step(metrics: Float, epoch: Option[Int]): Unit = {
    val current = metrics
    val current_epoch = epoch.getOrElse(last_epoch + 1)
    last_epoch = current_epoch

    if (is_better(current, best)) {
      best = current
      num_bad_epochs = 0
    } else {
      num_bad_epochs += 1
    }

    if (in_cooldown) {
      cooldown_counter -= 1
      num_bad_epochs = 0 // 忽略冷却期内的bad epochs
    }

    if (num_bad_epochs > patience) {
      _reduce_lr(current_epoch)
      cooldown_counter = cooldown
      num_bad_epochs = 0
    }

    _last_lr = optimizer.param_groups.map(param => param.paramGroup.options().get_lr().toFloat)
  }

  /** 降低学习率
    */
  private def _reduce_lr(epoch: Int): Unit = {
    for ((param_group, min_lr) <- optimizer.param_groups.zip(min_lrs)) {
      val old_lr = param_group.paramGroupDict("lr").asInstanceOf[Float]
      val new_lr = max(old_lr * factor, min_lr)
      if (old_lr - new_lr > eps) {
        param_group.paramGroupDict("lr") = new_lr
        param_group.paramGroup.options().set_lr(new_lr)
        if (verbose) {
          println(f"Epoch $epoch: reducing learning rate to $new_lr%.4e.")
        }
      }
    }
  }

  /** 检查是否处于冷却期
    */
  def in_cooldown: Boolean = {
    cooldown_counter > 0
  }

  /** 判断当前指标是否更好
    */
  def is_better(a: Float, best: Float): Boolean = {
    if (mode == "min" && threshold_mode == "rel") {
      val rel_epsilon = 1.0f - threshold
      a < best * rel_epsilon
    } else if (mode == "min" && threshold_mode == "abs") {
      a < best - threshold
    } else if (mode == "max" && threshold_mode == "rel") {
      val rel_epsilon = threshold + 1.0f
      a > best * rel_epsilon
    } else { // mode == "max" && threshold_mode == "abs"
      a > best + threshold
    }
  }

  /** 初始化is_better函数的参数
    */
  private def _init_is_better(mode: String, threshold: Float, threshold_mode: String): Unit = {
    require(mode == "min" || mode == "max", s"mode $mode is unknown!")
    require(
      threshold_mode == "rel" || threshold_mode == "abs",
      s"threshold mode $threshold_mode is unknown!"
    )

    if (mode == "min") {
      mode_worse = Float.PositiveInfinity
    } else { // mode == "max"
      mode_worse = Float.NegativeInfinity
    }
  }

  override def get_lr(): Seq[Float] = {
    throw new UnsupportedOperationException("get_lr() is not supported for reduce_lr_on_plateau")
  }

  override def state_dict(): Map[String, Any] = {
    super.state_dict() +
      ("best" -> best) +
      ("num_bad_epochs" -> num_bad_epochs) +
      ("cooldown_counter" -> cooldown_counter) +
      ("mode_worse" -> mode_worse)
  }

  override def load_state_dict(state_dict: Map[String, Any]): Unit = {
    super.load_state_dict(state_dict)
    best = state_dict.get("best").map(_.asInstanceOf[Float]).getOrElse(mode_worse)
    num_bad_epochs = state_dict.get("num_bad_epochs").map(_.asInstanceOf[Int]).getOrElse(0)
    cooldown_counter = state_dict.get("cooldown_counter").map(_.asInstanceOf[Int]).getOrElse(0)
    mode_worse = state_dict
      .get("mode_worse")
      .map(_.asInstanceOf[Float])
      .getOrElse(if (mode == "min") Float.PositiveInfinity else Float.NegativeInfinity)
  }
}
