package torch.utils.models.financial.fraudtransformer.layer

import torch.*
import torch.nn as nn
import torch.nn.{LayerNorm, functional as F}
import torch.optim.*
import torch.nn.modules.{HasParams, TensorModule}
import scala.math as Math
import scala.collection.mutable.ArrayBuffer

class TabTransformerEncoderBlock[ParamType <: FloatNN: Default](
    inputDim: Int,
    nHeads: Int,
    attDropout: Double,
    ffnMult: Int,
    ffnDropout: Double,
    ffnAct: String,
    anDropout: Double
) extends TensorModule[ParamType]
    with HasParams[ParamType] {

  val attention = MultiHeadAttention[ParamType](inputDim, nHeads, attDropout)
  val ffn = FeedForward[ParamType](inputDim, ffnMult, ffnDropout, ffnAct)
  val addNorm1 = AddNormConnection[ParamType](inputDim, anDropout)
  val addNorm2 = AddNormConnection[ParamType](inputDim, anDropout)

  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    val attOut = attention(x, x, x)
    val addNorm1Out = addNorm1(x, attOut)
    val ffnOut = ffn(addNorm1Out)
    addNorm2(addNorm1Out, ffnOut)
  }

  def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)
}
