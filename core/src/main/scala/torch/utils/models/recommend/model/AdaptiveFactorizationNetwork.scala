package torch.utils.models.recommend.model

import org.bytedeco.pytorch.GeneratorOptional
import torch.*
import torch.nn.modules.{HasParams, TensorModule}
import torch.utils.models.recommend.layer as nns
import torch.nn.functional as fn
import torch.utils.models.recommend.layer.{
  FeaturesEmbedding,
  FeaturesLinear,
  IntOrString,
  MultiLayerPerceptron
}

import scala.language.postfixOps

class LNN[ParamType <: FloatNN: Default](
    numFields: Int,
    embedDim: Int,
    LnnDim: Int,
    useBias: Boolean = false
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val lnn_output_dim = LnnDim * embedDim

  var weight = registerParameter(torch.zeros(Seq(LnnDim, numFields)), true, "weight")

  var bias = torch.zeros(Seq(LnnDim, embedDim)) // todo    LNN_dim, embed_dim is shape
  if (useBias) {
    registerParameter(bias, true, "bias")
  } else {
//    registerParameter("bias", null)
  }
  this.reset_parameters()

  def reset_parameters(): Unit = {
    val stdv = 1.0 / Math.sqrt(this.weight.size(1))

    val generatorOptional = new GeneratorOptional()
    this.weight.native.data.uniform_(-stdv, stdv, generatorOptional)
    if (this.bias != null) {
      this.bias.native.data.uniform_(-stdv, stdv, generatorOptional)
    }
  }

  def apply(xs: Tensor[ParamType]): Tensor[ParamType] = {

    val embed_x_abs = torch.abs(xs)
    val embed_x_afn = torch.add(embed_x_abs, torch.Tensor(1e-7))
    val embed_x_log = torch.log1p(embed_x_afn)
    val lnn_out = torch.matmul(this.weight, embed_x_log)
    if (this.bias != null) {
      lnn_out.add(this.bias)
    }
    val lnn_exp = torch.expm1(lnn_out)
    val output = fn.relu(lnn_exp).contiguous().view(-1, lnn_output_dim)
    output.to(xs.dtype)

  }

}

class AdaptiveFactorizationNetwork[ParamType <: FloatNN: Default](
    fieldDims: Seq[Int],
    embedDim: Int,
    LNNDim: Int,
    mlpDims: Seq[Int],
    dropouts: Seq[Float]
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val numFields = fieldDims.size
  val linear = register(FeaturesLinear(fieldDims))
  val embedding = register(FeaturesEmbedding(fieldDims, embedDim))
  val lnnOutputDim = LNNDim * embedDim
  val lnn = register(new LNN(numFields, embedDim, LNNDim))
  val mlp = register(MultiLayerPerceptron(lnnOutputDim, mlpDims, dropouts.head))

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    var x = input
    val embed_x = embedding(x)
    val lnn_out = lnn(embed_x)
    x = linear(x).add(mlp(lnn_out))
    torch.sigmoid(x.squeeze(1)).to(x.dtype)
  }
}

object AdaptiveFactorizationNetwork:
  def apply[ParamType <: FloatNN: Default](
      field_dims: Seq[Int],
      embed_dim: Int,
      LNN_dim: Int,
      mlp_dims: Seq[Int],
      dropouts: Seq[Float]
  ): AdaptiveFactorizationNetwork[ParamType] =
    new AdaptiveFactorizationNetwork(field_dims, embed_dim, LNN_dim, mlp_dims, dropouts)
