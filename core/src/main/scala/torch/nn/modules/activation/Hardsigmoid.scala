package torch
package nn
package modules
package activation

import org.bytedeco.pytorch
import torch.internal.NativeConverters.fromNative

final class Hardsigmoid[D <: DType: Default] extends TensorModule[D]:

  override def hasBias(): Boolean = false

  override def toString = getClass().getSimpleName()

  def apply(t: Tensor[D]): Tensor[D] = torch.hardsigmoid(t)

  def forward(input: Tensor[D]): Tensor[D] = torch.hardsigmoid(input)

object Hardsigmoid:
  def apply[D <: DType: Default](): Hardsigmoid[D] = new Hardsigmoid()
