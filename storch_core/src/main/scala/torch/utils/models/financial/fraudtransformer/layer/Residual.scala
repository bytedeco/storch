package torch.utils.models.financial.fraudtransformer.layer

import torch.*
import torch.nn as nn
import torch.nn.{LayerNorm, functional as F}
import torch.optim.*
import torch.nn.modules.{HasParams, TensorModule}
import scala.math as Math
import scala.collection.mutable.ArrayBuffer

class Residual[ParamType <: FloatNN: Default](fn: TensorModule[ParamType])
    extends TensorModule[ParamType]
    with HasParams[ParamType] {

  def forward(x: Tensor[ParamType], kwargs: Map[String, Any] = Map.empty): Tensor[ParamType] = {
    // 在Storch中处理可变参数可能需要不同的方法
    fn(x) + x
  }

  def apply(x: Tensor[ParamType], kwargs: Map[String, Any] = Map.empty): Tensor[ParamType] =
    forward(x, kwargs)

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???
}
