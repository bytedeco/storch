package torch
package optim
package lr_scheduler

//import torch.optim.Optimizer
import org.bytedeco.pytorch.ReduceLROnPlateauScheduler
import org.bytedeco.pytorch.LRScheduler as LRS

enum SchedulerMode(val value: Int):
  case min extends SchedulerMode(0)
  case max extends SchedulerMode(1)

  def intern(): SchedulerMode =
    SchedulerMode.values.find(_.value == this.value).getOrElse(this)

  def toNative: org.bytedeco.pytorch.ReduceLROnPlateauScheduler.SchedulerMode = this match
    case min => org.bytedeco.pytorch.ReduceLROnPlateauScheduler.SchedulerMode.min
    case max => org.bytedeco.pytorch.ReduceLROnPlateauScheduler.SchedulerMode.max

object SchedulerMode:
  def fromString(s: String): SchedulerMode = s match {
    case "min" | "Min" | "MIN" => min
    case "max" | "Max" | "MAX" => max
    case _                     => throw new IllegalArgumentException(s"无效的SchedulerMode: $s")
  }

object ThresholdMode:
  def fromString(s: String): ThresholdMode = s match {
    case "rel" | "Rel" | "REL" => rel
    case "abs" | "Abs" | "ABS" => abs
    case _                     => throw new IllegalArgumentException(s"无效的ThresholdMode: $s")
  }

//  override def toString: String = intern().

enum ThresholdMode(val value: Int):
  case rel extends ThresholdMode(0)
  case abs extends ThresholdMode(1)

  def toNative: org.bytedeco.pytorch.ReduceLROnPlateauScheduler.ThresholdMode = this match
    case rel => org.bytedeco.pytorch.ReduceLROnPlateauScheduler.ThresholdMode.rel
    case abs => org.bytedeco.pytorch.ReduceLROnPlateauScheduler.ThresholdMode.abs

  def intern(): ThresholdMode =
    ThresholdMode.values.find(_.value == this.value).getOrElse(this)

//  override def toString: String = intern().name
class TorchLRScheduler(optimizer: Optimizer) extends LRS(optimizer.native) {
  override def step() = super.step()
}

class ReduceLROnPlateauLR(
    optimizer: Optimizer,
    mode: String = "min",
    factor: Float = 0.1f,
    patience: Int = 10,
    threshold: Double = 1e-4,
    threshold_mode: String = "rel",
    cooldown: Int = 0,
    min_lr: Array[Float] = Array(0),
    eps: Double = 1e-8,
    verbose: Boolean = false
) extends ReduceLROnPlateauScheduler(
      optimizer.native,
      SchedulerMode.fromString(mode).toNative,
      factor,
      patience,
      threshold,
      ThresholdMode.fromString(threshold_mode).toNative,
      cooldown,
      min_lr,
      eps,
      verbose
    ) {
  override def step(metric: Float) = super.step(metric)
}
