package torch.utils.models.financial.fraudtransformer.layer

import torch.*
import torch.nn
import torch.nn.{BatchNorm1d, Dropout, LayerNorm, Linear, ReLU, Sequential, functional as F}
import torch.optim.*
import torch.nn.modules.{HasParams, TensorModule}
import scala.math as Math
import scala.collection.mutable.ArrayBuffer

class MultiLayerPerceptron[ParamType <: FloatNN: Default](
    inputDim: Int,
    layerDims: Array[Int],
    dropout: Double = 0.5,
    outputLayer: Boolean = true
) extends TensorModule[ParamType]
    with HasParams[ParamType] {

  val layers = ArrayBuffer[TensorModule[ParamType]]()
  var currentDim = inputDim

  for (layerDim <- layerDims) {
    layers += nn.Linear[ParamType](currentDim, layerDim)
    layers += nn.BatchNorm1d[ParamType](layerDim)
    layers += nn.ReLU[ParamType]()
    layers += nn.Dropout[ParamType](dropout.toFloat)
    currentDim = layerDim
  }

  if (outputLayer) {
    layers += nn.Linear[ParamType](currentDim, 1)
  }

  val mlp = nn.Sequential(layers.toSeq*)

  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    mlp(x)
  }

  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)
}
