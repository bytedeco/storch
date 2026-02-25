package torch.utils.models.recommend.layer

import torch.*
import torch.nn.modules.{HasParams, TensorModule}

import scala.collection.mutable.ArrayBuffer

class InnerProductNetwork[ParamType <: FloatNN: Default]()
    extends HasParams[ParamType]
    with TensorModule[ParamType] {

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
    val inputRow = input.index(::, row.toSeq)
    val inputCol = input.index(::, col.toSeq)
    val dotTensor = torch.dot(inputRow, inputCol).to(input.dtype)
    torch.sumWithDim(input = dotTensor, dim = 2, keepdim = false)
  }
}

object InnerProductNetwork:
  def apply[ParamType <: FloatNN: Default](): InnerProductNetwork[ParamType] =
    new InnerProductNetwork()
