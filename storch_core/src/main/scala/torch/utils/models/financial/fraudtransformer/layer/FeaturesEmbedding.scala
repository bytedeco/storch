package torch.utils.models.financial.fraudtransformer.layer

import torch.*
import torch.nn as nn
import torch.nn.{Dropout, Embedding, LayerNorm, functional as F}
import torch.optim.*
import torch.nn.modules.{HasParams, TensorModule}
import scala.math as Math
import scala.collection.mutable.ArrayBuffer
import org.bytedeco.pytorch.GeneratorOptional
class FeaturesEmbedding[ParamType <: FloatNN: Default](fieldDims: Array[Int], embedDim: Int)
    extends TensorModule[ParamType]
    with HasParams[ParamType] {

  val totalDims = fieldDims.sum
  val embedding = nn.Embedding[ParamType](totalDims, embedDim)

  // 计算偏移量
  val offsets = {
    val offs = ArrayBuffer[Long](0)
    var sum = 0L
    for (i <- 0 until fieldDims.length - 1) {
      sum += fieldDims(i)
      offs += sum
    }
    offs.toArray
  }

  // 初始化权重
  embedding.weight.data().normal_(0.0d, 0.01d) // , new GeneratorOptional()

  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    val xWithOffsets = x + torch.tensor(offsets).unsqueeze(0).to(this.paramType)
    embedding(xWithOffsets)
  }

  def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)
}
