package torch.utils.models.recommend.model

import org.bytedeco.pytorch.ModuleListImpl
import torch.*
import torch.nn.MultiheadAttention
import torch.nn.modules.{HasParams, TensorModule}
import torch.utils.models.recommend.layer as nns
import torch.utils.models.recommend.layer.IntOrString
import torch.nn.functional.{relu, sigmoid}
import torch.nn.modules.container.{ModuleList, Sequential}
class AutomaticFeatureInteractionModel[ParamType <: FloatNN: Default](
    fieldDims: Seq[Int],
    embedDim: Int,
    attenEmbedDim: Int,
    numHeads: Int,
    numLayers: Int,
    mlpDims: Seq[Int],
    dropouts: Seq[Float],
    hasResidual: Boolean = true
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val numFields = fieldDims.size
  val linear = register(nns.FeaturesLinear(fieldDims))
  val embedding = register(nns.FeaturesEmbedding(fieldDims, embedDim))
  val attenEmbedding = register(nn.Linear(embedDim, attenEmbedDim))
  val embedOutputDim = fieldDims.size * embedDim
  val attenOutputDim = fieldDims.size * attenEmbedDim
  val mlp = register(nns.MultiLayerPerceptron(embedOutputDim, mlpDims, dropouts(1)))
  val selfAttens = nn.ModuleList[ParamType]()
  for (i <- 0 until numLayers by 1) {
    val atten = nn.MultiheadAttention[ParamType](attenEmbedDim, numHeads, dropout = dropouts(0))
    selfAttens.insert(i, atten)
  }

  val attenFC = register(nn.Linear(attenOutputDim, 1))
  val V_res_embedding = register(nn.Linear(embedDim, attenEmbedDim))
//  val atten = register(nns.SelfAttention(attenOutputDim,numHeads,selfAttenList))
  val V_res = register(nn.Linear(attenOutputDim, embedOutputDim))
  val output = register(nns.MultiLayerPerceptron(embedOutputDim, mlpDims, dropouts(2)))

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    var x = input
    val embed_x = this.embedding(x)
    val atten_x = this.attenEmbedding(embed_x)
    var cross_term = atten_x.transpose(0, 1)
    var step = 0
    selfAttens.map(atten => {
      val cross_term_vec =
        atten.asInstanceOf[MultiheadAttention[ParamType]](cross_term, cross_term, cross_term)
      cross_term = cross_term_vec._1
      step += 1
    })
    cross_term = cross_term.transpose(0, 1)
    if (hasResidual) {
      val V_res_x = this.V_res_embedding(embed_x)
      cross_term.add(V_res_x)
    }
    cross_term = relu(cross_term).contiguous().view(-1, attenOutputDim)
    x = this.linear(x).add(this.attenFC(cross_term)).add(this.mlp(embed_x.view(-1, embedOutputDim)))
    torch.sigmoid(x.squeeze(1)).to(x.dtype)
  }

}

object AutomaticFeatureInteractionModel:
  def apply[ParamType <: FloatNN: Default](
      field_dims: Seq[Int],
      embed_dim: Int,
      atten_embed_dim: Int,
      num_heads: Int,
      num_layers: Int,
      mlp_dims: Seq[Int],
      dropouts: Seq[Float],
      has_residual: Boolean = true
  ): AutomaticFeatureInteractionModel[ParamType] =
    new AutomaticFeatureInteractionModel(
      field_dims,
      embed_dim,
      atten_embed_dim,
      num_heads,
      num_layers,
      mlp_dims,
      dropouts,
      has_residual
    )
