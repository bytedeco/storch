package torch.utils.models.llm.moe

import torch.nn.modules.{HasParams, TensorModule}
import torch.{Default, FloatNN, Tensor, nn}

class BasicExpert[ParamType <: FloatNN: Default](featureIn: Int, featureOut: Int)
    extends HasParams[ParamType]
    with TensorModule[ParamType] {
  val linear = register(nn.Linear(featureIn, featureOut))
  def apply(x: Tensor[ParamType]): Tensor[ParamType] = {
    linear.forward(x)
  }
  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    linear.forward(x)
  }
}
