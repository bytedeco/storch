package torch.utils.models.financial.fraudtransformer.layer

import torch.*
import torch.nn
import torch.nn.{LayerNorm, functional as F}
import torch.optim.*
import torch.nn.modules.{HasParams, TensorModule}
import scala.math as Math
import scala.collection.mutable.ArrayBuffer

class GEGLU[ParamType <: FloatNN: Default]
    extends TensorModule[ParamType]
    with HasParams[ParamType] {

  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    val xPart_gates = x.chunk(2, dim = -1)
    val xPart = xPart_gates(0)
    val gates = xPart_gates(1)
    xPart * F.gelu(gates)
  }

  def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)
}
