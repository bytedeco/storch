package torch
package nn
package modules
package activation

import org.bytedeco.pytorch
import torch.internal.NativeConverters.fromNative
//https://pytorch.org/docs/stable/generated/torch.lerp.html
final class Lerp[D <: DType: Default](val end: Tensor[D], val weight: Float = 0.5)
    extends TensorModule[D]:

  override def hasBias(): Boolean = false

  override def toString = getClass().getSimpleName()

  def apply(input: Tensor[D]): Tensor[D] = {
    torch.lerp(input, end, weight)
  }

  def forward(input: Tensor[D]): Tensor[D] = {
    torch.lerp(input, end, weight)
  }

object Lerp:
  def apply[D <: DType: Default](end: Tensor[D], weight: Float = 0.5): Lerp[D] =
    new Lerp(end, weight)

//start = torch.arange(1., 5.)
//end = torch.empty(4).fill_(10)
//start
//end
//torch.lerp(start, end, 0.5)
//torch.lerp(start, end, torch.full_like(start, 0.5))
