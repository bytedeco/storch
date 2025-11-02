package torch.utils.models.recommend.model

import torch.*
import torch.nn.modules.{HasParams, TensorModule}
import torch.utils.models.recommend.layer as nns
import torch.utils.models.recommend.layer.IntOrString

class DSSM[ParamType <: FloatNN: Default](
    userDatatypes: Seq[Map[String, IntOrString]],
    itemDatatypes: Seq[Map[String, IntOrString]],
    userDnnSize: (Int, Int) = (256, 128),
    itemDnnSize: (Int, Int) = (256, 128),
    dropout: Float = 0.0,
    activation: String = "ReLU",
    useSenet: Boolean = false
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val userTower = register(nns.Tower(userDatatypes, userDnnSize, dropout, activation, useSenet))
  val itemTower = register(nns.Tower(itemDatatypes, itemDnnSize, dropout, activation, useSenet))

  def apply(
      userFeat: Tensor[ParamType],
      itemFeat: Tensor[ParamType]
  ): (Tensor[ParamType], Tensor[ParamType]) = {
    (userTower(userFeat), itemTower(itemFeat))
  }

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = ???
}

object DSSM:
  def apply[ParamType <: FloatNN: Default](
      user_datatypes: Seq[Map[String, IntOrString]],
      item_datatypes: Seq[Map[String, IntOrString]],
      user_dnn_size: (Int, Int) = (256, 128),
      item_dnn_size: (Int, Int) = (256, 128),
      dropout: Float = 0.0,
      activation: String = "ReLU",
      use_senet: Boolean = false
  ): DSSM[ParamType] = new DSSM(
    user_datatypes,
    item_datatypes,
    user_dnn_size,
    item_dnn_size,
    dropout,
    activation,
    use_senet
  )
