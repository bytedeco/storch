package torch
package optim
package lr_scheduler

//import torch.optim.Optimizer

/** 通过指定的函数乘以学习率
  */
class MultiplicativeLR(
    override val optimizer: Optimizer,
    lr_lambda: Either[Int => Float, Seq[Int => Float]],
    last_epoch: Int = -1,
    verbose: Boolean = false
) extends LRScheduler(optimizer) {
//  var verbose: Boolean = verbose

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

    if (last_epoch > 0) {
      optimizer.param_groups.zip(lr_lambdas).map { case (group, lmbda) =>
        group.paramGroupDict("lr").asInstanceOf[Float] * lmbda(last_epoch)
      }
    } else {
      optimizer.param_groups.map(param => param.paramGroup.options().get_lr().toFloat)
    }
  }
}
