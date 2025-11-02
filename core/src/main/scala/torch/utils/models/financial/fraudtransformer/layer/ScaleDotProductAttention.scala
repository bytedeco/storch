package torch.utils.models.financial.fraudtransformer.layer

import torch.*
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.*
import scala.math as Math
import scala.collection.mutable.ArrayBuffer
import torch.nn.modules.{HasParams, TensorModule}

class ScaleDotProductAttention[ParamType <: FloatNN: Default](dropout: Double = 0.5)
    extends TensorModule[ParamType]
    with HasParams[ParamType] {

  val dropoutLayer = nn.Dropout[ParamType](dropout.toFloat)

  def forward(
      q: Tensor[ParamType],
      k: Tensor[ParamType],
      v: Tensor[ParamType],
      mask: Option[Tensor[ParamType]] = None
  ): Tensor[ParamType] = {
    val d = q.size(-1)
    val initAttn = q.matmul(k.transpose(2, 3)) / Math.sqrt(d)
    var attnScores: Tensor[ParamType] = initAttn.to(this.paramType)

    mask.foreach { m =>
      attnScores = attnScores.masked_fill(m.eq(0), -1e9).to(this.paramType)
    }

    attnScores = F.softmax(attnScores, dim = -1)
    attnScores = dropoutLayer(attnScores.to(this.paramType))
    attnScores.matmul(v).to(this.paramType)
  }

  def apply(
      q: Tensor[ParamType],
      k: Tensor[ParamType],
      v: Tensor[ParamType],
      mask: Option[Tensor[ParamType]] = None
  ): Tensor[ParamType] =
    forward(q, k, v, mask)

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???
}
