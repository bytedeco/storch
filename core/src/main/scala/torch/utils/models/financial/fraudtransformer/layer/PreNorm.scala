package torch.utils.models.financial.fraudtransformer.layer

import torch.*
import torch.nn as nn
import torch.nn.{LayerNorm, functional as F}
import torch.optim.*
import torch.nn.modules.{HasParams, TensorModule}
import scala.math as Math
import scala.collection.mutable.ArrayBuffer

class PreNorm[ParamType <: FloatNN: Default](dim: Int, fn: TensorModule[ParamType])
    extends TensorModule[ParamType]
    with HasParams[ParamType] {

  val norm = nn.LayerNorm[ParamType](Seq(dim))

  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    fn(norm(x))
  }

  def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)
}
