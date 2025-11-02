package torch.utils.models.recommend.layer

import torch.*
import torch.nn.functional as F
import torch.nn.modules.{HasParams, TensorModule}

import scala.collection.mutable.ArrayBuffer

class FieldAwareFactorizationMachine[ParamType <: FloatNN: Default](
    field_dims: Seq[Int],
    embed_dim: Int
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val num_fields = field_dims.length
  val embeddings =
    nn.ModuleList[ParamType]() // field_dims.map(i => register(nn.Embedding(i, embed_dim))))
  val offsets = field_dims.scanLeft(0L)((ep, et) => ep + et).dropRight(1).map(_.toLong).toArray
  val embModule = nn.FMEmbedding(field_dims.sum.toInt, embed_dim.toInt)
  // todo modulelist append has bigs bug
//  Range(1, this.num_fields).foreach(i => embeddings.append(register(embModule)))
  Range(1, this.num_fields).foreach(i =>
    embeddings.insert(i, register(embModule, "embeddings[" + i + "]"))
  )

  def init_weight(): Unit = {}

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {

    input.add(input.new_tensor(offsets).unsqueeze(0).expand(input.size(0), num_fields - 1))
    var xs = Range(0, num_fields, 1).map(i => embeddings(i)(input)).toArray
    val ix = ArrayBuffer[Tensor[ParamType]]()
    for (i <- 0 until num_fields - 1) {
      for (j <- i + 1 until num_fields) {
        val xsj = xs(j).index(::, i)
        val xsi = xs(i).index(::, j)
        ix.append(torch.dot(xsj, xsi).to(input.dtype))

      }
    }
    val out = torch.stack(ix.toSeq, dim = 1)
    out

  }
}

object FieldAwareFactorizationMachine:
  def apply[ParamType <: FloatNN: Default](
      field_dims: Seq[Int],
      embed_dim: Int
  ): FieldAwareFactorizationMachine[ParamType] =
    new FieldAwareFactorizationMachine(field_dims, embed_dim)
