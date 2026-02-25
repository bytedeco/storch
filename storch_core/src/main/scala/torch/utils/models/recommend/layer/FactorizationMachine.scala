package torch.utils.models.recommend.layer

import torch.*
import torch.nn.modules.{HasParams, TensorModule}

/** *
  *
  * @param reduce_sum
  * @param default$ParamType$0
  * @tparam ParamType
  *   ix.mul(x.to(dtype = this.paramType))
  */
class FactorizationMachine[ParamType <: FloatNN: Default](reduce_sum: Boolean = true)
    extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val reduceSum: Boolean = reduce_sum

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {

    val squareOfSum = torch.pow(torch.sumWithDim(input, dim = 1), 2)
    val sumOfSquare = torch.sumWithDim(torch.pow(input, 2), dim = 1)
    val ix = squareOfSum.sub(sumOfSquare)
    if (reduceSum) torch.sum(ix) else ix
    val x = torch.Tensor(0.5)
    val res = ix.multiply(x.to(dtype = this.paramType))
    res

  }

}

object FactorizationMachine:
  def apply[ParamType <: FloatNN: Default](
      reduce_sum: Boolean = true
  ): FactorizationMachine[ParamType] = new FactorizationMachine(reduce_sum)
