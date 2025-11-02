package torch
package optim
package lr_scheduler

//import torch.optim.Optimizer

/** 链式调用多个学习率调度器
  */
class ChainedScheduler(
    schedulers: Seq[LRScheduler]
) extends LRScheduler(schedulers.head.optimizer) {
  require(schedulers.nonEmpty, "At least one scheduler must be provided")
  require(
    schedulers.forall(_.optimizer == schedulers.head.optimizer),
    "chained_scheduler expects all schedulers to belong to the same optimizer"
  )

  val _schedulers: Seq[LRScheduler] = schedulers
  override val optimizer: Optimizer = schedulers.head.optimizer
  _last_lr = _schedulers.last.optimizer.param_groups.map(param =>
    param.paramGroup.options().get_lr().toFloat
  )
//  _last_lr = _schedulers.last.optimizer.param_groups.map(_("lr").asInstanceOf[Float])

  override def get_lr(): Seq[Float] = {
    throw new UnsupportedOperationException("get_lr() is not supported for chained_scheduler")
  }

  override def step(epoch: Option[Int] = None): Unit = {
    // 对每个调度器执行step
    for (scheduler <- _schedulers) {
      scheduler.step(epoch)
    }
//optimizer.param_groups.map(param => param.options().get_lr().toFloat)
    _last_lr = _schedulers.last.optimizer.param_groups.map(param =>
      param.paramGroup.options().get_lr().toFloat
    )
//    _last_lr = _schedulers.last.optimizer.param_groups.map(_("lr").asInstanceOf[Float])
  }

  override def state_dict(): Map[String, Any] = {
    // 保存所有调度器的状态
    val state = super.state_dict()
    val scheduler_states = _schedulers.map(_.state_dict())
    state + ("_schedulers" -> scheduler_states)
  }

  override def load_state_dict(state_dict: Map[String, Any]): Unit = {
    super.load_state_dict(state_dict)
    val scheduler_states = state_dict("_schedulers").asInstanceOf[Seq[Map[String, Any]]]

    for ((scheduler, state) <- _schedulers.zip(scheduler_states)) {
      scheduler.load_state_dict(state)
    }
  }
}
