package torch.utils.models.financial.fraudtransformer.layer

import torch.*
import torch.nn as nn
import torch.nn.{Dropout, LayerNorm, Sequential, functional as F}
import torch.optim.*
import torch.nn.modules.{HasParams, TensorModule}
import scala.math as Math
import scala.collection.mutable.ArrayBuffer

class TabTransformerEncoder[ParamType <: FloatNN: Default](
    inputDim: Int,
    depth: Int,
    nHeads: Int,
    attDropout: Double,
    ffnMult: Int,
    ffnDropout: Double,
    ffnAct: String,
    anDropout: Double
) extends TensorModule[ParamType]
    with HasParams[ParamType] {

  val transformerBlocks = (0 until depth).map { _ =>
    TabTransformerEncoderBlock[ParamType](
      inputDim,
      nHeads,
      attDropout,
      ffnMult,
      ffnDropout,
      ffnAct,
      anDropout
    )
  }

  val sequential = nn.Sequential(transformerBlocks*)

  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    sequential(x)
  }

  def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)
}
