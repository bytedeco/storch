package torch.utils.models.llm.gpt

import torch.*
import torch.nn.modules.container.{ModuleDict, ModuleList, Sequential}
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.{LayerNorm, functional as F}

class Block[ParamType <: FloatNN: Default](config: GPTConfig)
    extends HasParams[ParamType]
    with TensorModule[ParamType] {
  val head_size = config.n_embd / config.n_head
  val att: MultiHeadAttention[ParamType] = MultiHeadAttention(config)
  val ffn: FeedForward[ParamType] = FeedForward(config)
  val ln1: LayerNorm[ParamType] = nn.LayerNorm(Seq(config.n_embd))
  val ln2: LayerNorm[ParamType] = nn.LayerNorm(Seq(config.n_embd))

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)

  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {

    println(s"Debug: Block input shape ${x.shape.mkString(", ")}")
    val attenOut = att.forward(ln1(x))
    println(s"Debug: Block attention output shape ${attenOut.shape.mkString(", ")}")
    val x1 = x + attenOut
    x1 + ffn.forward(ln2(x1))
  }
}
