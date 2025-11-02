package torch.utils.models.recommend.layer

import torch.*
import torch.nn.modules.{HasParams, TensorModule}

class MultiLayerPerceptron[ParamType <: FloatNN: Default](
    input_dim: Int,
    embed_dims: Seq[Int],
    dropout: Float,
    output_layer: Boolean = true
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  // todo  runtime  layers size is empty  is bug please fixup
  val layers = nn.ModuleList[ParamType]()
  // todo RuntimeException: Submodule 'MultiLayerPerceptron' already defined asued by ModuleList append has bigs bug

  embed_dims.zipWithIndex.foreach {
    case (embed_dim, index) => {
      val block =
        register(LinearBnReluDropBlock(input_dim, embed_dim, dropout), s"block_${index + 1}")
//      val block = LinearBnReluDropBlock(input_dim, embed_dim, dropout) //,s"block_${index+1}")
      layers.append(block)
//      layers.insert(index,register(LinearBnReluDropBlock(input_dim, embed_dim, dropout),s"block_${index+1}"))
    }
  }
  if (output_layer) {
//    var output_linear = register(nn.Linear(input_dim, 1))
    layers.insert(embed_dims.length, register(nn.Linear(input_dim, 1)))
  }

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {

    layers(input)

  }

}

object MultiLayerPerceptron:
  def apply[ParamType <: FloatNN: Default](
      input_dim: Int,
      embed_dims: Seq[Int],
      dropout: Float,
      output_layer: Boolean = true
  ): MultiLayerPerceptron[ParamType] =
    new MultiLayerPerceptron(input_dim, embed_dims, dropout, output_layer)

//val mlp = nn.Sequential[ParamType]()

//    val moduleDict = nn.ModuleDict[ParamType]()
//          moduleDict.insert("block"+index, block)

//  register(LinearBnReluDropBlock(input_dim, 10, dropout), s"block_${1}")
//  register(LinearBnReluDropBlock(input_dim, 10, dropout), s"block_${2}")
//  register(LinearBnReluDropBlock(input_dim, 10, dropout), s"block_${3}")
//  layers.append(register(LinearBnReluDropBlock(input_dim, 10, dropout),s"block_${1}"))
//  layers.append(register(LinearBnReluDropBlock(input_dim, 10, dropout),s"block_${2}"))
//  layers.append(register(LinearBnReluDropBlock(input_dim, 10, dropout),s"block_${3}"))
