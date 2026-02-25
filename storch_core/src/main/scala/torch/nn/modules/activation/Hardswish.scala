//package torch.nn.modules.activation
package torch
package nn
package modules
package activation

import org.bytedeco.pytorch
import torch.internal.NativeConverters.fromNative

final class Hardswish[D <: DType: Default] extends TensorModule[D]:

  override def hasBias(): Boolean = false

  override def toString = getClass().getSimpleName()

  def apply(t: Tensor[D]): Tensor[D] = torch.hardswish(t)

  def forward(input: Tensor[D]): Tensor[D] = torch.hardswish(input)

object Hardswish:
  def apply[D <: DType: Default](): Hardswish[D] = new Hardswish()
