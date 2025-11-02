package torch.utils.models.recommend.model

import torch.*
import torch.nn.modules.container.{ModuleDict, ModuleList, Sequential}
import torch.nn.modules.{HasParams, TensorModule}
import torch.utils.models.recommend.layer as nns
import torch.utils.models.recommend.layer.IntOrString

import scala.collection.mutable.ListBuffer

final class SharedBottomModel[ParamType <: FloatNN: Default](
    categorical_field_dims: Seq[Int],
    numerical_num: Int,
    embed_dim: Int,
    bottom_mlp_dims: Seq[Int],
    tower_mlp_dims: Seq[Int],
    task_num: Int,
    dropout: Float
) extends HasParams[ParamType]
    with TensorModule[ParamType] {
  val embedding = registerModule(
    nns.FeaturesEmbedding(categorical_field_dims, embed_dim = embed_dim)
  )
  val numerical_layer = registerModule(nn.Linear(numerical_num, embed_dim))
  val embed_output_dim = (categorical_field_dims.length + 1) * embed_dim

  val bottom = registerModule(
    nns.MultiLayerPerceptron(embed_output_dim, bottom_mlp_dims, dropout, false)
  )
  val towerLayers =
    for i <- 0 until (task_num)
    yield nns.MultiLayerPerceptron(bottom_mlp_dims.last, tower_mlp_dims, dropout)
  val tower = ModuleList[ParamType](towerLayers*)

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???

  def apply(
      categorical_x: Tensor[ParamType],
      numerical_x: Tensor[ParamType]
  ): Seq[Tensor[ParamType]] = {
    val categorical_emb = embedding(categorical_x)
    val numerical_emb = numerical_layer(numerical_x).unsqueeze(1)
    val emb = torch.cat(Seq(categorical_emb, numerical_emb), dim = 1).view(-1, embed_output_dim)
    val fea = bottom(emb)
    val results =
      for i <- 0 until (task_num) yield torch.sigmoid(tower(i)(fea).squeeze(1)).to(this.paramType)
    results.toSeq
  }

}

object SharedBottomModel {
  def apply[ParamType <: FloatNN: Default](
      categorical_field_dims: Seq[Int],
      numerical_num: Int,
      embed_dim: Int,
      bottom_mlp_dims: Seq[Int],
      tower_mlp_dims: Seq[Int],
      task_num: Int,
      dropout: Float
  ): SharedBottomModel[ParamType] = new SharedBottomModel(
    categorical_field_dims,
    numerical_num,
    embed_dim,
    bottom_mlp_dims,
    tower_mlp_dims,
    task_num,
    dropout
  )

}
