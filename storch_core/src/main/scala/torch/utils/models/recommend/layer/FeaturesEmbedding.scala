package torch.utils.models.recommend.layer

import torch.nn.modules.{HasWeight, TensorModule}
import torch.*

import scala.annotation.nowarn

class FeaturesEmbedding[ParamType <: FloatNN | ComplexNN: Default](
    fieldDims: Seq[Int],
    embedDim: Int = 1
) extends TensorModule[ParamType] {

  val embedding = register(nn.Embedding(fieldDims.sum, embedDim))
  var offsets: Array[Long] =
    fieldDims.map(_.toInt).scanLeft(0)((ep, et) => ep + et).dropRight(1).map(_.toLong).toArray
  init_weights(embedding)

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    val offsetsTensor = torch.Tensor(offsets).unsqueeze(0)
    input.add(offsetsTensor)
    embedding(input)
//    embedding(input.to(dtype = torch.int64))
  }

  @nowarn("msg=unused private member")
  private def init_weights(m: HasWeight[ParamType]): Unit = {
    torch.nn.init.xavierUniform_(m.weight)
  }
}

object FeaturesEmbedding:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      field_dims: Seq[Int],
      embed_dim: Int = 1
  ): FeaturesEmbedding[ParamType] = new FeaturesEmbedding(field_dims, embed_dim)
