package torch.utils.models.recommend.model

import torch.*
import torch.nn.modules.container.{ModuleList, Sequential}
import torch.nn.modules.{HasParams, TensorModule}
import torch.utils.models.recommend.layer as nns
import torch.utils.models.recommend.layer.IntOrString

import scala.collection.mutable.ListBuffer

final class PLEModel[ParamType <: FloatNN: Default](
    categorical_field_dims: Seq[Int],
    numerical_num: Int,
    embed_dim: Int,
    bottom_mlp_dims: Seq[Int],
    tower_mlp_dims: Seq[Int],
    task_num: Int,
    shared_expert_num: Int,
    specific_expert_num: Int,
    dropout: Float
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val embedding = registerModule(
    nns.FeaturesEmbedding(categorical_field_dims, embed_dim = embed_dim)
  )
  val numerical_layer = registerModule(nn.Linear(numerical_num, embed_dim))
  val embedOutputDim = (categorical_field_dims.length + 1) * embed_dim
  val layersNum = bottom_mlp_dims.length
  val taskNum = task_num
  val specificExpertNum = specific_expert_num
  val sharedExpertNum = shared_expert_num
  val bottomMlpDims = bottom_mlp_dims
  val towerMlpDims = tower_mlp_dims
  val embedDim = embed_dim
  private val taskExperts: ModuleList[ParamType] = nn.ModuleList(
    (0 until layersNum).map { i =>
      nn.ModuleList(
        (0 until taskNum).map { _ =>
          nn.ModuleList(
            (0 until specificExpertNum).map { _ =>
              val inputDim = if (i == 0) embedOutputDim else bottomMlpDims(i - 1)
              nns.MultiLayerPerceptron(
                inputDim,
                List(bottomMlpDims(i)),
                dropout,
                output_layer = false
              )
            }*
          )
        }*
      )
    }*
  )

  private val taskGates: ModuleList[ParamType] = nn.ModuleList(
    (0 until layersNum).map { i =>
      nn.ModuleList(
        (0 until taskNum).map { _ =>
          val inputDim = if (i == 0) embedOutputDim else bottomMlpDims(i - 1)
          nn.Sequential(
            nn.Linear(inputDim, sharedExpertNum + specificExpertNum),
            nn.Softmax(dim = 1)
          )
        }*
      )
    }*
  )

  private val shareExperts: ModuleList[ParamType] = nn.ModuleList(
    (0 until layersNum).map { i =>
      nn.ModuleList(
        (0 until sharedExpertNum).map { _ =>
          val inputDim = if (i == 0) embedOutputDim else bottomMlpDims(i - 1)
          nns.MultiLayerPerceptron(inputDim, List(bottomMlpDims(i)), dropout, output_layer = false)
        }*
      )
    }*
  )

  private val shareGates: ModuleList[ParamType] = nn.ModuleList(
    (0 until layersNum).map { i =>
      val inputDim = if (i == 0) embedOutputDim else bottomMlpDims(i - 1)
      nn.Sequential(
        nn.Linear(inputDim, sharedExpertNum + taskNum * specificExpertNum),
        nn.Softmax(dim = 1)
      )
    }*
  )

  private val tower: ModuleList[ParamType] = nn.ModuleList(
    (0 until taskNum).map { _ =>
      nns.MultiLayerPerceptron(bottomMlpDims.last, towerMlpDims, dropout)
    }*
  )

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???

  def apply(
      categorical_x: Tensor[ParamType],
      numerical_x: Tensor[ParamType]
  ): Seq[Tensor[ParamType]] = {
    val categorical_emb = embedding(categorical_x)
    val numerical_emb = numerical_layer(numerical_x).unsqueeze(1)
    val emb = torch.cat(Seq(categorical_emb, numerical_emb), dim = 1).view(-1, embedOutputDim)
    val task_feaSeq = for i <- 0 until (task_num + 1) yield emb
    val task_fea = new ListBuffer[Tensor[ParamType]]()
    task_fea.addAll(task_feaSeq)
    for (i <- 0 until (layersNum)) {
      val share_output = shareExperts(i)
        .asInstanceOf[ModuleList[ParamType]]
        .map(expert => expert(task_fea(-1)).unsqueeze(1))
      val taskOutputList = new ListBuffer[Tensor[ParamType]]()
      for (j <- 0 until (task_num)) {
        val taskExpertsCell: ModuleList[ParamType] =
          taskExperts(i).asInstanceOf[ModuleList[ParamType]]
        val task_output = taskExpertsCell(j)
          .asInstanceOf[ModuleList[ParamType]]
          .map(expert => expert(task_fea(j)).unsqueeze(1))
        taskOutputList.appendedAll(task_output)
        val mix_output = torch.cat(taskOutputList.toList ++ share_output, dim = 1)
        val gate_value =
          taskGates(i).asInstanceOf[ModuleList[ParamType]](j)(task_fea(j)).unsqueeze(1)
        task_fea(j) = torch.bmm(gate_value, mix_output).squeeze(1)
      }
      if (i != layersNum - 1) {
        val gate_value = shareGates(i)(task_fea(-1)).unsqueeze(1)
        taskOutputList.appendedAll(share_output)
        val mix_output = torch.cat(taskOutputList.toSeq, dim = 1)
        task_fea(-1) = torch.bmm(gate_value, mix_output).squeeze(1)
      }
    }
    val results = for i <- 0 until (task_num) yield torch.sigmoid(tower(i)(task_fea(i)).squeeze(1))
    results.map(_.to(this.paramType))
  }
}

object PLEModel {
  def apply[ParamType <: FloatNN: Default](
      categorical_field_dims: Seq[Int],
      numerical_num: Int,
      embed_dim: Int,
      bottom_mlp_dims: Seq[Int],
      tower_mlp_dims: Seq[Int],
      task_num: Int,
      shared_expert_num: Int,
      specific_expert_num: Int,
      dropout: Float
  ): PLEModel[ParamType] = new PLEModel(
    categorical_field_dims,
    numerical_num,
    embed_dim,
    bottom_mlp_dims,
    tower_mlp_dims,
    task_num,
    shared_expert_num,
    specific_expert_num,
    dropout
  )

}

//  val task_experts = for i <- 0 until (layers_num) yield task_num
//  val task_gates = for i <- 0 until (layers_num) yield task_num
//  val share_experts = layers_num
//  val share_gates = layers_num
//  for (i <- 0 until (layers_num)) {
//    val input_dim = if i == 0 then embed_output_dim else bottom_mlp_dims(i - 1)
//    val shareExpertsLayers = for i <- 0 until (shared_expert_num) yield nns.MultiLayerPerceptron(input_dim, Seq(bottom_mlp_dims(i)), dropout, false)
//    share_experts(i) = ModuleList[ParamType](shareExpertsLayers *)
//    share_gates(i) = Sequential[ParamType](nn.Linear(input_dim, shared_expert_num + task_num * specific_expert_num),
//      nn.Softmax(dim = 1))
//    for (j <- 0 until (task_num)) {
//      val taskExpertsLayer = for k <- 0 until (specific_expert_num) yield nns.MultiLayerPerceptron(input_dim, Seq(bottom_mlp_dims(i)), dropout, false)
//      task_experts(i)(j) = ModuleList[ParamType](taskExpertsLayer *)
//      task_gates(i)(j) = Sequential[ParamType](
//        nn.Linear(input_dim, shared_expert_num + specific_expert_num),
//        nn.Softmax(dim = 1)
//      )
//    }
//    task_experts(i) = ModuleList[ParamType](task_experts(i))
//    task_gates(i) = ModuleList[ParamType](task_gates(i))
//  }
//
