package torch.utils.models.recommend.model

import torch.*
import torch.nn.modules.container.{ModuleDict, ModuleList, Sequential}
import torch.nn.modules.{HasParams, TensorModule}
import torch.utils.models.recommend.layer as nns
import torch.utils.models.recommend.layer.IntOrString

final class OMoEModel[ParamType <: FloatNN: Default](
    categorical_field_dims: Seq[Int],
    numerical_num: Int,
    embed_dim: Int,
    bottom_mlp_dims: Seq[Int],
    tower_mlp_dims: Seq[Int],
    task_num: Int,
    expert_num: Int,
    dropout: Float
) extends HasParams[ParamType]
    with TensorModule[ParamType] {
  val embedding = registerModule(
    nns.FeaturesEmbedding(categorical_field_dims, embed_dim = embed_dim)
  )
  val numerical_layer = registerModule(nn.Linear(numerical_num, embed_dim))
  val embed_output_dim = (categorical_field_dims.length + 1) * embed_dim

  val expertLayers =
    for i <- 0 until (expert_num)
    yield nns.MultiLayerPerceptron(embed_output_dim, bottom_mlp_dims, dropout, output_layer = false)
  val expert = ModuleList[ParamType](expertLayers*)
  val towerLayers =
    for i <- 0 until (task_num)
    yield nns.MultiLayerPerceptron(bottom_mlp_dims.last, tower_mlp_dims, dropout)
  val tower = ModuleList[ParamType](towerLayers*)
  val gate = Sequential[ParamType](
    nn.Linear(embed_output_dim, expert_num),
    nn.Softmax(dim = 1)
  )

  def apply(
      categorical_x: Tensor[ParamType],
      numerical_x: Tensor[ParamType]
  ): Seq[Tensor[ParamType]] = {
    val categorical_emb = embedding(categorical_x)
    val numerical_emb = numerical_layer(numerical_x).unsqueeze(1)
    val emb = torch.cat(Seq(categorical_emb, numerical_emb), dim = 1).view(-1, embed_output_dim)
    val gate_value = gate(emb).unsqueeze(1)
    val feaLayer = for i <- 0 until (expert_num) yield expert(i)(emb).unsqueeze(1)
    var fea = torch.cat(feaLayer, dim = 1)
    fea = torch.bmm(gate_value, fea).squeeze(1)
    val results = for i <- 0 until (task_num) yield torch.sigmoid(tower(i)(fea).squeeze(1))
    results.map(_.to(this.paramType))
  }

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???
}

object OMoEModel {
  def apply[ParamType <: FloatNN: Default](
      categorical_field_dims: Seq[Int],
      numerical_num: Int,
      embed_dim: Int,
      bottom_mlp_dims: Seq[Int],
      tower_mlp_dims: Seq[Int],
      task_num: Int,
      expert_num: Int,
      dropout: Float
  ): OMoEModel[ParamType] = new OMoEModel(
    categorical_field_dims,
    numerical_num,
    embed_dim,
    bottom_mlp_dims,
    tower_mlp_dims,
    task_num,
    expert_num,
    dropout
  )

}
