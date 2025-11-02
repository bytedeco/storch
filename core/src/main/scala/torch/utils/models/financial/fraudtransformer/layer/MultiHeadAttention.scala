package torch.utils.models.financial.fraudtransformer.layer

import torch.*
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.*
import scala.math as Math
import scala.collection.mutable.ArrayBuffer
import torch.nn.modules.{HasParams, TensorModule}

class MultiHeadAttention[ParamType <: FloatNN: Default](
    inputDim: Int,
    nHead: Int,
    dropout: Double = 0.5
) extends TensorModule[ParamType]
    with HasParams[ParamType] {

  val headDim = inputDim / nHead

  val qW = nn.Linear[ParamType](inputDim, nHead * headDim, bias = false)
  val kW = nn.Linear[ParamType](inputDim, nHead * headDim, bias = false)
  val vW = nn.Linear[ParamType](inputDim, nHead * headDim, bias = false)
  val fc = nn.Linear[ParamType](nHead * headDim, inputDim, bias = false)
  val attention = ScaleDotProductAttention[ParamType](dropout)

  def forward(
      q: Tensor[ParamType],
      k: Tensor[ParamType],
      v: Tensor[ParamType],
      mask: Option[Tensor[ParamType]] = None
  ): Tensor[ParamType] = {
    val batchSize = q.size(0)
    val seqLen = q.size(1)

    val qTrans = qW(q).view(batchSize, seqLen, nHead, headDim).transpose(1, 2)
    val kTrans = kW(k).view(batchSize, seqLen, nHead, headDim).transpose(1, 2)
    val vTrans = vW(v).view(batchSize, seqLen, nHead, headDim).transpose(1, 2)

    val masked = mask.map(_.unsqueeze(1))
    val attnOut = attention(qTrans, kTrans, vTrans, masked)
    val attnTrans = attnOut.transpose(1, 2).contiguous().view(batchSize, seqLen, -1)

    fc(attnTrans)
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
