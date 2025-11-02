package torch.utils.models.recommend.layer

import torch.*
import torch.nn.functional as F
import torch.nn.modules.{HasParams, TensorModule}

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.language.postfixOps
class AttentionalFactorizationMachine[ParamType <: FloatNN: Default](
    embed_dim: Long,
    attn_size: Long,
    dropouts: Seq[Float],
    training: Boolean = true
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val attention = register(nn.Linear(embed_dim, attn_size))
  val projection = register(nn.Linear(attn_size, 1))
  val fc = register(nn.Linear(embed_dim, 1))

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    val inputShape = input.shape
    val num_fields = inputShape(1)
    val row = new ArrayBuffer[Long]()
    val col = new ArrayBuffer[Long]()
    for (i <- 0 until num_fields - 1 by 1) {
      for (j <- i + 1 until num_fields by 1) {
        row.append(i)
        col.append(j)
      }
    }
    val p = input.index(::, row.toSeq)
    val q = input.index(::, col.toSeq)
    val innerProduct = torch.dot(p, q) // p * q
    var attn_scores = F.relu(attention(innerProduct))
    attn_scores = F.softmax(projection(attn_scores), dim = 1)
    attn_scores = F.dropout(attn_scores, p = dropouts(0), training = training)
    val scoreProduct = torch.dot(attn_scores, innerProduct).to(input.dtype)
    var attn_output = torch.sumWithDim(input = scoreProduct, dim = 1)
    attn_output = F.dropout(attn_output, p = dropouts(1), training = training)
    fc(attn_output)

  }
}

object AttentionalFactorizationMachine:
  def apply[ParamType <: FloatNN: Default](
      embed_dim: Long,
      attn_size: Long,
      dropouts: Seq[Float],
      training: Boolean = true
  ): AttentionalFactorizationMachine[ParamType] =
    new AttentionalFactorizationMachine(embed_dim, attn_size, dropouts, training)
