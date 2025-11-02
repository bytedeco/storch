package torch.utils.models.recommend.layer

import torch.*
import torch.nn.modules.{HasParams, TensorModule}

class Tower[ParamType <: FloatNN: Default](
    datatypes: Seq[Map[String, IntOrString]],
    dnnSize: (Int, Int) = (256, 128),
    dropout: Float = 0.0,
    activation: String = "ReLU",
    useSenet: Boolean = false
) extends HasParams[ParamType]
    with TensorModule[ParamType] {
  val dnns = nn.ModuleList[ParamType]()
  val embeddings = register(new EmbeddingModule(datatypes, useSenet))
  val activationLayer = activation match {
    case "relu" | "ReLU"           => nn.ReLU[ParamType]()
    case "tanh" | "Tanh"           => nn.Tanh[ParamType]()
    case "sigmoid" | "Sigmoid"     => nn.Sigmoid[ParamType]()
    case "leakyrelu" | "LeakyReLU" => nn.LeakyReLU[ParamType](0.2)
  }
  var inputDims = embeddings.sparseDim + embeddings.denseDim
  for (dim <- Seq(dnnSize._1, dnnSize._2)) {
    dnns.append(nn.Linear(inputDims, dim))
    dnns.append(nn.Dropout(dropout))
    dnns.append(activationLayer)
    inputDims = dim
  }

  def apply(x: Tensor[ParamType]): Tensor[ParamType] = {
    var dnnInput = embeddings(x)
    for (dnn <- dnns) {
      dnnInput = dnn(dnnInput)
    }
    dnnInput
  }

}

object Tower:
  def apply[ParamType <: FloatNN: Default](
      datatypes: Seq[Map[String, IntOrString]],
      dnn_size: (Int, Int) = (256, 128),
      dropout: Float = 0.0,
      activation: String = "ReLU",
      use_senet: Boolean = false
  ): Tower[ParamType] = new Tower(datatypes, dnn_size, dropout, activation, use_senet)
