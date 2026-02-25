package torch.utils.models.recommend.layer

import torch.*
import torch.nn.modules.{HasParams, TensorModule}

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

class CrossNetwork[ParamType <: FloatNN: Default](input_dim: Long, num_layers: Int)
    extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val w = nn.ModuleList[ParamType]()
  val b = ListBuffer[Tensor[ParamType]]() // nn.ModuleList[ParamType]()
  // todo Submodule 'CrossNetwork' already defined, cause by ModuleList append has bigs bug
  for (i <- 0 until num_layers) {
    w.insert(i, register(nn.Linear(input_dim, 1L, bias = false), s"block_${i}"))
//    w.append(register(nn.Linear(input_dim, 1l,add_bias = false)))
    b.append(torch.zeros(input_dim.toInt).to(dtype = this.paramType))
  }

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    val x0 = input
    var x = input
    for (i <- 0 until num_layers) {
      val xw = w(i)(x)
//        x0.*(xw).add(b(i).)
      x = torch.dot(x0, xw).add(b(i)).add(x)

    }
    x
  }
}

object CrossNetwork:
  def apply[ParamType <: FloatNN: Default](
      input_dim: Long,
      num_layers: Int
  ): CrossNetwork[ParamType] = new CrossNetwork(input_dim, num_layers)
