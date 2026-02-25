package torch.utils.models.financial.fraudtransformer.layer

import torch.*
import torch.nn as nn
import torch.nn.{Dropout, LayerNorm, functional as F}
import torch.optim.*
import torch.nn.modules.{HasParams, TensorModule}
import scala.math as Math
import scala.collection.mutable.ArrayBuffer

class AddNormConnection[ParamType <: FloatNN: Default](dim: Int, dropout: Double)
    extends TensorModule[ParamType]
    with HasParams[ParamType] {

  val layerNorm = nn.LayerNorm[ParamType](Seq(dim))
  val dropoutLayer = nn.Dropout[ParamType](dropout.toFloat)

  def forward(x: Tensor[ParamType], layerOut: Tensor[ParamType]): Tensor[ParamType] = {
    val xUpdated = x + dropoutLayer(layerOut)
    layerNorm(xUpdated)
  }

  def apply(x: Tensor[ParamType], layerOut: Tensor[ParamType]): Tensor[ParamType] =
    forward(x, layerOut)

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???
}
