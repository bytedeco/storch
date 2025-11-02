package torch
package nn
package modules
package activation

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{ELUOptions, SELUImpl, SELUOptions, SoftmaxOptions}
import torch.internal.NativeConverters.fromNative
import torch.*
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.modules.container.{ModuleDict, ModuleList, Sequential}
import scala.collection.mutable.ListBuffer

/** Applies the Softmax function to an n-dimensional input Tensor rescaling them so that the
  * elements of the n-dimensional output Tensor lie in the range [0,1] and sum to 1.
  *
  * Softmax is defined as: $$\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$
  *
  * When the input Tensor is a sparse tensor then the unspecifed values are treated as ``-inf``.
  */
final class GEGLU[D <: FloatNN: Default](val dim: Int = -1) extends TensorModule[D]:

  val linearLayer = register(nn.Linear(dim, dim * 2))

  override def hasBias(): Boolean = false

  override def toString =
    getClass().getSimpleName() + s"(dim=$dim)"

  def apply(input: Tensor[D]): Tensor[D] = {
    val xProj = linearLayer(input)
    val xPartGate = torch.chunk(xProj, 2, dim = -1)
    xPartGate(0) * torch.gelu(xPartGate(1))
  }

  def forward(input: Tensor[D]): Tensor[D] = {
    val xProj = linearLayer(input)
    val xPartGate = torch.chunk(xProj, 2, dim = -1)
    xPartGate(0) * torch.gelu(xPartGate(1))
  }

object GEGLU {

  def apply[D <: FloatNN: Default](dim: Int = -1): GEGLU[D] = new GEGLU(dim)
}

//import torch
//import torch.nn as nn
//
//class GEGLU(nn.Module):
//    def __init__(self, dim):
//        super().__init__()
//        self.linear = nn.Linear(dim, dim * 2)  # 双倍维度用于分割
//
//    def forward(self, x):
//        x_proj = self.linear(x)
//        x_part, gate = x_proj.chunk(2, dim=-1)  # 分割为两部分
//        return x_part * torch.nn.functional.gelu(gate)  # 门控融合
