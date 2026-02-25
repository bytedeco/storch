package torch.utils.models.llm.gpt

import torch.*
import torch.nn.functional as F
import torch.nn.modules.container.{ModuleDict, ModuleList, Sequential}
import torch.nn.modules.{HasParams, TensorModule}

class SingleHeadAttention[ParamType <: FloatNN: Default](config: GPTConfig)
    extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val key = register(nn.Linear(config.n_embd, config.head_size))
  val value = register(nn.Linear(config.n_embd, config.head_size))
  val query = register(nn.Linear(config.n_embd, config.head_size))
  val dropoutLayer = register(nn.Dropout(config.dropout))
  val head_size = config.head_size
  val attentionMask = register_buffer(
    "attention_mask",
    torch.tril(torch.ones(Seq(config.block_size, config.block_size)))
  )

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)

  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    val (batchSize, seqLen, hiddenSize) = (x.shape(0).toInt, x.shape(1).toInt, x.shape(2).toInt)
    val k = key(x)
    val v = value(x)
    val q = query(x)
    // compute the attention weights
    var weight = q.matmul(k.transpose(-2, -1))
    // mask the attention weights
    val mask = attentionMask(0.::(seqLen), 0.::(seqLen))
    weight = weight.masked_fill(mask == 0, Double.NegativeInfinity).to(x.dtype)
    weight = (weight / math.sqrt(config.head_size.toDouble).toFloat).to(x.dtype)
    // apply softmax and dropout
    weight = F.softmax(weight, -1)
    weight = dropoutLayer(weight)
    val out = weight.matmul(v)
    out
  }
}
