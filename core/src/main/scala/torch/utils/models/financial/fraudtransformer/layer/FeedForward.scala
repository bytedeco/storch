package torch.utils.models.financial.fraudtransformer.layer

import torch.*
import torch.nn as nn
import torch.nn.{Dropout, LayerNorm, Linear, ReLU, Sequential, functional as F}
import torch.optim.*
import torch.nn.modules.{HasParams, TensorModule}
import scala.math as Math
import scala.collection.mutable.ArrayBuffer

class FeedForward[ParamType <: FloatNN: Default](
    inputDim: Int,
    mult: Int = 4,
    dropout: Double = 0.5,
    ffAct: String = "GEGLU"
) extends TensorModule[ParamType]
    with HasParams[ParamType] {

  val net: TensorModule[ParamType] = {
    if (ffAct == "GEGLU") {
      nn.Sequential(
        nn.Linear[ParamType](inputDim, inputDim * mult * 2),
        nn.GEGLU[ParamType](),
        nn.Dropout[ParamType](dropout.toFloat),
        nn.Linear[ParamType](inputDim * mult, inputDim)
      )
    } else {
      nn.Sequential(
        nn.Linear[ParamType](inputDim, inputDim * mult),
        nn.ReLU[ParamType](),
        nn.Dropout[ParamType](dropout.toFloat),
        nn.Linear[ParamType](inputDim * mult, inputDim)
      )
    }
  }

  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    net(x)
  }

  def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)
}
