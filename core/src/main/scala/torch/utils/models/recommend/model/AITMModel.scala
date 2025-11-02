package torch.utils.models.recommend.model

import torch.*
import torch.nn.modules.container.{ModuleDict, ModuleList, Sequential}
import torch.nn.modules.{HasParams, TensorModule}
import torch.utils.models.recommend.layer as nns
import torch.utils.models.recommend.layer
import torch.utils.models.recommend.layer.{FeaturesEmbedding, IntOrString, MultiLayerPerceptron}

import scala.collection.mutable.ListBuffer

final class AITMModel[ParamType <: FloatNN: Default](
    categorical_field_dims: Seq[Int],
    numerical_num: Int,
    embed_dim: Int,
    bottom_mlp_dims: Seq[Int],
    tower_mlp_dims: Seq[Int],
    task_num: Int,
    dropout: Float
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val embedding = registerModule(FeaturesEmbedding(categorical_field_dims, embed_dim = embed_dim))
  val numerical_layer = registerModule(nn.Linear(numerical_num, embed_dim))
  val embed_output_dim = (categorical_field_dims.length + 1) * embed_dim
  val hidden_dim = bottom_mlp_dims.last
  val gLayers =
    for i <- 0 until (task_num - 1) yield nn.Linear(bottom_mlp_dims.last, bottom_mlp_dims.last)
  val gate = ModuleList[ParamType](gLayers*)
  val h1 = registerModule(nn.Linear(bottom_mlp_dims.last, bottom_mlp_dims.last))
  val h2 = registerModule(nn.Linear(bottom_mlp_dims.last, bottom_mlp_dims.last))
  val h3 = registerModule(nn.Linear(bottom_mlp_dims.last, bottom_mlp_dims.last))
  val bottomLayers =
    for i <- 0 until (task_num)
    yield MultiLayerPerceptron(embed_output_dim, bottom_mlp_dims, dropout, false)
  val bottom = ModuleList[ParamType](bottomLayers*)
  val towerLayers =
    for i <- 0 until (task_num)
    yield layer.MultiLayerPerceptron(bottom_mlp_dims.last, tower_mlp_dims, dropout)
  val tower = ModuleList[ParamType](towerLayers*)

  def apply(
      categorical_x: Tensor[ParamType],
      numerical_x: Tensor[ParamType]
  ): Seq[Tensor[ParamType]] = {
    val categorical_emb = embedding(categorical_x)
    val numerical_emb = numerical_layer(numerical_x).unsqueeze(1)
    val emb = torch.cat(Seq(categorical_emb, numerical_emb), dim = 1).view(-1, embed_output_dim)
    val feaLayer = for i <- 0 until (task_num) yield bottom(i)(emb)
    val fea = new ListBuffer[Tensor[ParamType]]()
    fea.addAll(feaLayer)
    for (i <- 1 until (task_num)) {
      val p = gate(i - 1)(fea(i - 1)).unsqueeze(1)
      val q = fea(i).unsqueeze(1)
      val x = torch.cat(Seq(p, q), dim = 1)
      val V = h1(x)
      val K = h2(x)
      val Q = h3(x)
//      fea[i] = torch.sum(torch.nn.functional.softmax(torch.sum(K * Q, 2, True) / np.sqrt(self.hidden_dim), dim=1) * V, 1)
      val softmax = nn.functional.softmax(
        torch.sum(K * Q, 2, true) / torch.sqrt(torch.Tensor(hidden_dim)),
        dim = 1
      ) * V
      fea(i) = torch.sumWithDim(softmax.to(this.paramType), dim = 1)
    }
    val results = for i <- 0 until (task_num) yield torch.sigmoid(tower(i)(fea(i).squeeze(1)))
    results.map(_.to(this.paramType))

  }

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???
}

object AITMModel:
  def apply[ParamType <: FloatNN: Default](
      categorical_field_dims: Seq[Int],
      numerical_num: Int,
      embed_dim: Int,
      bottom_mlp_dims: Seq[Int],
      tower_mlp_dims: Seq[Int],
      task_num: Int,
      dropout: Float
  ): AITMModel[ParamType] =
    new AITMModel(
      categorical_field_dims,
      numerical_num,
      embed_dim,
      bottom_mlp_dims,
      tower_mlp_dims,
      task_num,
      dropout
    )
