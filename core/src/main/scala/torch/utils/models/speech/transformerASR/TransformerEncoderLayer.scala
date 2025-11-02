package torch.utils.models.speech.transformerASR

import torch.*
import torch.nn.functional as F
import torch.nn.modules.{HasParams, TensorModule}

// Transformer编码器层实现
class TransformerEncoderLayer[ParamType <: FloatNN: Default](
    val d_model: Int,
    val nhead: Int,
    val dim_feedforward: Int,
    val dropout: Double
) extends HasParams[ParamType]
    with TensorModule[ParamType]:

  val self_attn: nn.MultiheadAttention[ParamType] =
    nn.MultiheadAttention(d_model, nhead, dropout = dropout)
  val linear1: nn.Linear[ParamType] = nn.Linear(d_model, dim_feedforward)
  val dropout1: nn.Dropout[ParamType] = nn.Dropout(dropout)
  val linear2: nn.Linear[ParamType] = nn.Linear(dim_feedforward, d_model)
  val dropout2: nn.Dropout[ParamType] = nn.Dropout(dropout)
  val norm1: nn.LayerNorm[ParamType] = nn.LayerNorm(Array(d_model))
  val norm2: nn.LayerNorm[ParamType] = nn.LayerNorm(Array(d_model))

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???
  // 前向传播
  def forward(
      src: Tensor[ParamType],
      src_mask: Option[Tensor[ParamType]] = None,
      src_key_padding_mask: Option[Tensor[ParamType]] = None
  ): Tensor[ParamType] =

    val src2 = self_attn(
      src,
      src,
      src,
      attn_mask = src_mask.get,
      key_padding_mask = src_key_padding_mask.get,
      need_weights = false,
      average_attn_weights = false
    )._1
    var output = src + dropout1(src2)
    output = norm1(output)
    val ffOutput = linear2(dropout2(F.relu(linear1(output))))
    output = output + dropout2(ffOutput)
    output = norm2(output)
    output

  // apply方法，默认调用forward
  def apply(
      src: Tensor[ParamType],
      src_mask: Option[Tensor[ParamType]] = None,
      src_key_padding_mask: Option[Tensor[ParamType]] = None
  ): Tensor[ParamType] = forward(src, src_mask, src_key_padding_mask)

  // 伴生对象，提供便捷的构造方法
object TransformerEncoderLayer:
  def apply[ParamType <: FloatNN: Default](
      d_model: Int,
      nhead: Int,
      dim_feedforward: Int = 2048,
      dropout: Double = 0.1
  ): TransformerEncoderLayer[ParamType] =
    new TransformerEncoderLayer[ParamType](d_model, nhead, dim_feedforward, dropout)
