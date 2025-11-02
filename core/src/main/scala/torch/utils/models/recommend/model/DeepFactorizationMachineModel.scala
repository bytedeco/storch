package torch.utils.models.recommend.model

import torch.*
import torch.nn.modules.{HasParams, TensorModule}
import torch.utils.models.recommend.layer as nn

class DeepFactorizationMachineModel[ParamType <: FloatNN: Default](
    fieldDims: Seq[Int],
    embedDim: Int,
    mlpDims: Seq[Int],
    dropout: Float
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val linear = register(nn.FeaturesLinear(fieldDims))
  val fm = register(nn.FactorizationMachine(reduce_sum = true))
  val embedding = register(nn.FeaturesEmbedding(fieldDims, embedDim))
  val embed_output_dim = fieldDims.size * embedDim
  val mlp = register(nn.MultiLayerPerceptron(embed_output_dim, mlpDims, dropout))

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    val embed_x = embedding(input)
    val linearOut = linear(input)
    val fmOut = fm(embed_x).add(mlp(embed_x.view(-1, embed_output_dim)))
    val linearOut1 = linear(input.to(dtype = this.paramType))
    val fmOut1 = fm(embed_x.to(dtype = this.paramType)).add(mlp(embed_x.view(-1, embed_output_dim)))
    val x = torch.add(linearOut, fmOut)
    torch.sigmoid(x.squeeze(1)).to(dtype = this.paramType)

  }
}

object DeepFactorizationMachineModel:
  def apply[ParamType <: FloatNN: Default](
      field_dims: Seq[Int],
      embed_dim: Int,
      mlp_dims: Seq[Int],
      dropout: Float
  ): DeepFactorizationMachineModel[ParamType] =
    new DeepFactorizationMachineModel(field_dims, embed_dim, mlp_dims, dropout)

//    val out = torch.sigmoid(x.squeeze(1).to(dtype = this.paramType))
//    out.to(dtype = this.paramType)
