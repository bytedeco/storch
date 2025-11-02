package torch.utils.models.recommend.layer

import torch.*
import torch.nn.modules.{HasParams, TensorModule}

import scala.math.{ceil, floor}

class SENet[ParamType <: FloatNN: Default](inputDim: Int, reduction: Int = 2)
    extends HasParams[ParamType]
    with TensorModule[ParamType] {
  require(reduction > 0, "reduction must be greater than 0")
  require(inputDim > 0, "input_dim must be greater than 0")
  val nets = nn.Sequential(
    nn.Linear(inputDim, floor(inputDim / reduction).toLong, false),
    nn.ReLU(),
    nn.Linear(floor(inputDim / reduction).toLong, inputDim, false),
    nn.Sigmoid()
  )

  def apply(x: Tensor[ParamType]): Tensor[ParamType] = {
    nets(x)
  }
}

object SENet:
  def apply[ParamType <: FloatNN: Default](input_dim: Int, reduction: Int = 2): SENet[ParamType] =
    new SENet(input_dim, reduction)
