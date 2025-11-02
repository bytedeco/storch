package torch.utils.models.recommend.model

import torch.*
import torch.nn.modules.container.{ModuleDict, ModuleList, Sequential}
import torch.nn.modules.{HasParams, TensorModule}
import torch.utils.models.recommend.layer as nns

import scala.collection.mutable.ListBuffer

final class SingleTaskModel[ParamType <: FloatNN: Default](
    categorical_field_dims: Seq[Int],
    numerical_num: Int,
    embed_dim: Int,
    bottom_mlp_dims: Seq[Int],
    tower_mlp_dims: Seq[Int],
    task_num: Int,
    dropout: Float
) extends HasParams[ParamType]
    with TensorModule[ParamType] {
  val embeddingLayers =
    for i <- 0 until (task_num)
    yield nns.FeaturesEmbedding(categorical_field_dims, embed_dim = embed_dim)
  val numericLayers = for i <- 0 until (task_num) yield nn.Linear(numerical_num, embed_dim)
  val embedding = ModuleList[ParamType](embeddingLayers*)
  val numerical_layer = ModuleList[ParamType](numericLayers*)
  val embed_output_dim = (categorical_field_dims.length + 1) * embed_dim
  val bottomLayers =
    for i <- 0 until (task_num)
    yield nns.MultiLayerPerceptron(embed_output_dim, bottom_mlp_dims, dropout, false)
  val bottom = ModuleList[ParamType](bottomLayers*)
  val towerLayers =
    for i <- 0 until (task_num)
    yield nns.MultiLayerPerceptron(bottom_mlp_dims.last, tower_mlp_dims, dropout)
  val tower = ModuleList[ParamType](towerLayers*)

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???

  def apply(
      categorical_x: Tensor[ParamType],
      numerical_x: Tensor[ParamType]
  ): Seq[Tensor[ParamType]] = {
    val results = new ListBuffer[Tensor[ParamType]]()
    for (i <- 0 until (task_num)) {
      val categorical_emb = embedding(i)(categorical_x)
      val numerical_emb = numerical_layer(i)(numerical_x).unsqueeze(1)
      val emb = torch.cat(Seq(categorical_emb, numerical_emb), dim = 1).view(-1, embed_output_dim)
      val fea = bottom(i)(emb)
      results.appended(torch.sigmoid(tower(i)(fea).squeeze(1)))
    }
    results.toSeq
  }
}

object SingleTaskModel {
  def apply[ParamType <: FloatNN: Default](
      categorical_field_dims: Seq[Int],
      numerical_num: Int,
      embed_dim: Int,
      bottom_mlp_dims: Seq[Int],
      tower_mlp_dims: Seq[Int],
      task_num: Int,
      dropout: Float
  ): SingleTaskModel[ParamType] = new SingleTaskModel(
    categorical_field_dims,
    numerical_num,
    embed_dim,
    bottom_mlp_dims,
    tower_mlp_dims,
    task_num,
    dropout
  )

}
