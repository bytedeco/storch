package torch
package optim
package lr_scheduler

//import torch.optim.Optimizer

/** 顺序调用一系列调度器
  */
class SequentialLR(
    override val optimizer: Optimizer,
    schedulers: Seq[LRScheduler],
    milestones: Seq[Int],
    last_epoch: Int = -1,
    verbose: Boolean = false
) extends LRScheduler(optimizer) {
//  override var verbose: Boolean = verbose

  // 验证调度器和里程碑参数
  require(schedulers.nonEmpty, "At least one scheduler must be provided")
  require(
    schedulers.forall(_.optimizer == optimizer),
    "All schedulers must belong to the same optimizer"
  )
  require(
    milestones.length == schedulers.length - 1,
    s"Sequential Schedulers expects number of schedulers provided to be one more than the number of milestone points, but got number of schedulers ${schedulers.length} and the number of milestones to be equal to ${milestones.length}"
  )

  val _schedulers: Seq[LRScheduler] = schedulers
  val _milestones: Seq[Int] = milestones.sorted
  this.last_epoch = last_epoch + 1 // 调整初始epoch

  // 重置学习率为初始值
  for (group <- optimizer.param_groups) {
    if (group.paramGroupDict.contains("initial_lr")) {
      group.paramGroupDict("lr") = group.paramGroupDict("initial_lr")
      group.paramGroup.options().set_lr(group.paramGroupDict("initial_lr").asInstanceOf[Double])
    }
  }

  // "撤销"其他调度器执行的步骤
  for (scheduler <- _schedulers) {
    scheduler.last_epoch -= 1
  }

  // 仅对第一个调度器执行初始步骤
  _schedulers.head._initial_step()

  _last_lr = _schedulers.head.get_last_lr()

  override def get_lr(): Seq[Float] = {
    throw new UnsupportedOperationException("get_lr() is not supported for sequential_lr")
  }

  override def step(epoch: Option[Int] = None): Unit = {
    // 忽略传入的epoch参数，使用内部的last_epoch
    last_epoch += 1

    // 找到当前应该使用的调度器
    val idx = _milestones.indexWhere(milestone => milestone > last_epoch)
    val scheduler = if (idx == -1) {
      _schedulers.last
    } else {
      _schedulers(idx)
    }

    // 如果刚进入一个新的调度器阶段，重置其epoch为0
    if (idx > 0 && _milestones(idx - 1) == last_epoch) {
      scheduler.step(Some(0))
    } else {
      scheduler.step()
    }

    _last_lr = scheduler.get_last_lr()
  }

  override def state_dict(): Map[String, Any] = {
    // 保存调度器状态
    val state = super.state_dict()
    val scheduler_states = _schedulers.map(_.state_dict())
    state + ("_schedulers" -> scheduler_states, "_milestones" -> _milestones)
  }

  override def load_state_dict(state_dict: Map[String, Any]): Unit = {
    super.load_state_dict(state_dict)
    val scheduler_states = state_dict("_schedulers").asInstanceOf[Seq[Map[String, Any]]]
    val milestones = state_dict("_milestones").asInstanceOf[Seq[Int]]

    for ((scheduler, state) <- _schedulers.zip(scheduler_states)) {
      scheduler.load_state_dict(state)
    }
  }
}
