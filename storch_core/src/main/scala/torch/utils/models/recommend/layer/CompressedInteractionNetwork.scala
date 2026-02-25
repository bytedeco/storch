package torch.utils.models.recommend.layer

import torch.*
import torch.nn.functional as F
import torch.nn.modules.{HasParams, TensorModule}

import scala.collection.mutable.ArrayBuffer

class CompressedInteractionNetwork[ParamType <: FloatNN: Default](
    input_dim: Int,
    cross_layer_sizes: Seq[Int],
    split_half: Boolean = true
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val num_layers = cross_layer_sizes.length
  val splitHalf = split_half
  var prev_dim: Int = 1 // 0 //todo  why 0 or 1
  var fc_input_dim: Long = 0
  val conv_layers = nn.ModuleList[ParamType]()
  for (i <- 0 until num_layers) {
    var cross_layer_size = cross_layer_sizes(i)
    // todo moduleList append has bigs bug
    conv_layers.insert(
      i,
      nn.Conv1d(
        input_dim * prev_dim,
        cross_layer_size,
        kernel_size = 1,
        stride = 1,
        dilation = 1,
        bias = true
      )
    )
    if (splitHalf && i != num_layers - 1) {
      cross_layer_size = math.floor(cross_layer_size / 2).toInt
    }
    prev_dim = cross_layer_size
    fc_input_dim += prev_dim

  }
  val fc = register(nn.Linear(fc_input_dim, 1))

  def apply(rawInput: Tensor[ParamType]): Tensor[ParamType] = {

    var input = rawInput
    val x0 = input.unsqueeze(2)
    var h = input
    val xs = new ArrayBuffer[Tensor[ParamType]]()
    for (i <- 0 until num_layers) {
      input = torch.dot(x0, h.unsqueeze(1))
      val batch_size = input.shape(0)
      val f0_dim = input.shape(1)
      val fin_dim = input.shape(2)
      val embed_dim = input.shape(3)
      input = input.view(batch_size, f0_dim * fin_dim, embed_dim)
      input = F.relu(conv_layers(i)(input))
      if (splitHalf && i != num_layers - 1) {
        val input_h = torch.split(input, math.floor(input.shape(1) / 2).toInt, dim = 1)
        input = input_h(0)
        val h = input_h(1)

      } else {
        h = input
      }
      xs.append(input)
    }
    fc(torch.sum(torch.cat(xs.toSeq, dim = 1), 2))
  }
}

object CompressedInteractionNetwork:
  def apply[ParamType <: FloatNN: Default](
      input_dim: Int,
      cross_layer_sizes: Seq[Int],
      split_half: Boolean = true
  ): CompressedInteractionNetwork[ParamType] =
    new CompressedInteractionNetwork(input_dim, cross_layer_sizes, split_half)

//    conv_layers.append(
//      nn.Conv1d(
//        input_dim * prev_dim,
//        cross_layer_size,
//        kernel_size = 1,
//        stride = 1,
//        dilation = 1,
//        bias = true
//      )
