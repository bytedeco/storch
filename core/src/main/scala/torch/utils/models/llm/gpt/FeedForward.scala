package torch.utils.models.llm.gpt

import torch.*
import torch.nn.functional as F
import torch.nn.modules.container.{ModuleDict, ModuleList, Sequential}
import torch.nn.modules.{HasParams, TensorModule}

class FeedForward[ParamType <: FloatNN: Default](config: GPTConfig) extends HasParams[ParamType] {

  val net: Sequential[ParamType] = nn.Sequential(
    nn.Linear(config.n_embd, 4 * config.n_embd),
    nn.GELU(),
    nn.Linear(4 * config.n_embd, config.n_embd),
    nn.Dropout(config.dropout)
  )
  def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)

  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    net(x)
  }
}
