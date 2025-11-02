package torch
package optim
package lr_scheduler

//import torch.optim.Optimizer

/** 在指定的里程碑处将学习率乘以gamma因子
  */
class MultiStepLR(
    override val optimizer: Optimizer,
    milestones: Seq[Int],
    gamma: Float = 0.1f,
    last_epoch: Int = -1,
    verbose: Boolean = false
) extends LRScheduler(optimizer)
    with ClosedFormLR {
//  override var verbose: Boolean = verbose
//  val milestones: Seq[Int] = milestones.sorted
//  val gamma: Float = gamma
  // 计算每个epoch的gamma指数
  private val milestone_counts: Map[Int, Int] =
    milestones.sorted.groupBy(identity).mapValues(_.length).toMap

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

    if (!milestone_counts.contains(last_epoch)) {
      optimizer.param_groups.map(param => param.paramGroup.options().get_lr().toFloat)
    } else {
      optimizer.param_groups.map(group =>
        group.paramGroupDict("lr").asInstanceOf[Float] * math
          .pow(gamma, milestone_counts(last_epoch))
          .toFloat
      )
    }
  }

  override def get_closed_form_lr(): Seq[Float] = {
    // 计算last_epoch之前的里程碑数量
    val count = milestones.sorted.count(_ <= last_epoch)
    base_lrs.map(base_lr => base_lr * math.pow(gamma, count).toFloat)
  }
}
