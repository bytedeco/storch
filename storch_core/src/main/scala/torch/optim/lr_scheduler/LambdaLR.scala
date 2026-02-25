package torch
package optim
package lr_scheduler

//import torch.optim.Optimizer

/** 根据给定函数调整学习率
  */
class LambdaLR(
    override val optimizer: Optimizer,
    lr_lambda: Either[Int => Float, Seq[Int => Float]],
    last_epoch: Int = -1,
    verbose: Boolean = false
) extends LRScheduler(optimizer) {
//  override var verbose: Boolean = verbose

  // 处理lr_lambda参数
  val lr_lambdas: Seq[Int => Float] = lr_lambda match {
    case Left(f) => Seq.fill(optimizer.param_groups.length)(f)
    case Right(seq) =>
      require(
        seq.length == optimizer.param_groups.length,
        s"Expected ${optimizer.param_groups.length} lr_lambdas, but got ${seq.length}"
      )
      seq
  }

  // 初始化
  if (last_epoch == -1) {
    // 设置初始学习率
    for (group <- optimizer.param_groups) {
      if (!group.paramGroupDict.contains("initial_lr")) {
        group.paramGroupDict("initial_lr") = group.paramGroup.options().get_lr() // "lr")
      }
    }
  } else {
    // 检查是否存在initial_lr
    for ((group, i) <- optimizer.param_groups.zipWithIndex) {
      require(
        group.paramGroupDict.contains("initial_lr"),
        s"param 'initial_lr' is not specified in param_groups[$i] when resuming an optimizer"
      )
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

    base_lrs.zip(lr_lambdas).map { case (base_lr, lmbda) => base_lr * lmbda(last_epoch) }
  }

  override def state_dict(): Map[String, Any] = {
    // 在Scala中无法直接保存函数，这里仅保存基本状态
    super.state_dict()
  }
}
