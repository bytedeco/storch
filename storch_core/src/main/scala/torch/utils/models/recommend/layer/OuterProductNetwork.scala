package torch.utils.models.recommend.layer

import torch.*
import torch.nn.functional as F
import torch.nn.modules.{HasParams, TensorModule}

import scala.collection.mutable.ArrayBuffer
class OuterProductNetwork[ParamType <: FloatNN: Default](
    num_fields: Long,
    embed_dim: Long,
    kernel_type: String = "mat"
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val num_ix = Math.floorDiv(num_fields * (num_fields - 1), 2)
  var kernel_shape = new ArrayBuffer[Long]()
  if (kernel_type == "mat") {

    kernel_shape += embed_dim
    kernel_shape += num_ix
    kernel_shape += embed_dim
  } else if (kernel_type == "vec") {
    kernel_shape += num_ix
    kernel_shape += embed_dim
  } else if (kernel_type == "num") {
    kernel_shape += num_ix
    kernel_shape += 1
  } else {
    throw new IllegalArgumentException("kernel_type must be mat or vec")
  }
  val kernel = registerParameter(
    nn.init.xavierUniform_(torch.zeros(kernel_shape.map(_.toInt).toSeq))
  )

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {

    val num_fields = input.shape(1)
    val row = ArrayBuffer[Long]()
    val col = ArrayBuffer[Long]()
    for (i <- 0 until num_fields - 1) {
      for (j <- i + 1 until num_fields) {
        row += i
        col += j
      }
    }
    val p = input.index(::, row.toArray)
    val q = input.index(::, col.toArray)
    if (kernel_type == "mat") {
      val dotUnsqueeze = torch.dot(p.unsqueeze(1), kernel).to(input.dtype)
      val kp = torch.sum(input = dotUnsqueeze, dim = 1).permute(0, 2, 1)
      val kpqDot = torch.dot(kp, q)
      torch.sumWithDim(input = kpqDot, dim = 1)
    } else {
      val pq = torch.dot(p, q)
      val kpq = torch.dot(pq, kernel.unsqueeze(0)).to(input.dtype)

      torch.sumWithDim(input = kpq, dim = -1)
    }
  }
}

object OuterProductNetwork:
  def apply[ParamType <: FloatNN: Default](
      num_fields: Long,
      embed_dim: Long,
      kernel_type: String = "mat"
  ): OuterProductNetwork[ParamType] = new OuterProductNetwork(num_fields, embed_dim, kernel_type)
