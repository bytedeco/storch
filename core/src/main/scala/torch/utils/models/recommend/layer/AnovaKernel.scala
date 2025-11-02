package torch.utils.models.recommend.layer

import torch.*
import torch.nn.modules.{HasParams, TensorModule}

class AnovaKernel[ParamType <: FloatNN: Default](order: Int, reduce_sum: Boolean = true)
    extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val reduceSum: Boolean = reduce_sum

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    val shape = input.shape
    val batch_size = shape.head
    val num_fields = shape(1)
    val embed_dim = shape(2)
    var a_prev = torch.ones(Seq(batch_size, num_fields + 1, embed_dim))
    var a = torch.zeros(Seq(batch_size, num_fields + 1, embed_dim))
    for (t <- 0 until order by 1) {
      a.index(::, (t + 1).&&(0), ::)
        .+=(torch.mul(input.index(::, t.&&(0), ::), a_prev.index(::, t.&&(-1), ::)))
      a = torch.cumsum(a, dim = 1)
      a_prev = a
    }
    if (reduceSum) {
      torch
        .sum(a.index(::, -1, ::), dim = Seq(1), keepdim = false, dtype = this.paramType)
        .to(dtype = this.paramType)
    } else {
      a.index(::, -1, ::).to(dtype = this.paramType)
    }

  }

}

object AnovaKernel:
  def apply[ParamType <: FloatNN: Default](
      order: Int,
      reduce_sum: Boolean = true
  ): AnovaKernel[ParamType] = new AnovaKernel(order, reduce_sum)

//      torch.sum(a.index(::,-1,::).to(dtype = this.paramType),dim = Seq(1),keepdim = false,dtype = this.paramType)
//    val squareOfSum = torch.pow(torch.sum(input,dim=1),2)
//    val sumOfSquare = torch.sum(torch.pow(input,2),dim=1)
//    val ix = squareOfSum.sub(sumOfSquare)
//    if(reduceSum) torch.sum(ix) else ix
//    val x = torch.Tensor(0.5)
//    ix.multiply(x.to(dtype = this.paramType))
