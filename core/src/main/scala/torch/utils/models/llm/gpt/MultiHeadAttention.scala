package torch.utils.models.llm.gpt

import torch.*
import torch.nn.modules.container.{ModuleDict, ModuleList, Sequential}
import torch.nn.modules.regularization.Dropout
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.{Linear, functional as F}

class MultiHeadAttention[ParamType <: FloatNN: Default](config: GPTConfig)
    extends HasParams[ParamType] {

  val heads = nn.ModuleList((1 to config.n_head).map(conf => new SingleHeadAttention(config))*)
  val proj = register(nn.Linear(config.n_embd * config.n_head, config.n_embd))
  val dropoutLayer = register(nn.Dropout(config.dropout))

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)

  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    // compute the output of each head and concatenate them
    val headOutputs = heads.map(attn => attn(x))
    println(
      s"Debug: MultiHeadAttention: single head output shape: ${headOutputs.head.shape.mkString(", ")}"
    )
    val concatenated = torch.cat(headOutputs.toSeq, -1)
    println(s"Debug: MultiHeadAttention: concatenated shape: ${concatenated.shape.mkString(", ")}")
    val output = proj(concatenated)
    dropoutLayer(output)
  }

}
