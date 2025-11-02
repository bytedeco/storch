package torch
package optim
package lr_scheduler

//import torch.optim.Optimizer
import org.bytedeco.pytorch.StepLR

/** 每隔指定的epoch数将学习率乘以gamma因子
  */
class StepScheduler(
    optimizer: Optimizer,
    step_size: Int,
    gamma: Double = 0.1d,
    last_epoch: Int = -1,
    verbose: Boolean = false
) extends StepLR(optimizer.native, step_size, gamma) {

//  def step() = super.step()
//  override def get_lr(): Seq[Float] = {

}
