package torch.utils.data

import torch.{DType, Default, Tensor}

trait TensorDataset {

  def length: Long

  def getItem(idx: Int): Tensor[? <: DType]

}

object TensorDataset {

  def apply[Input <: DType, Target <: DType](
      _features: Tensor[Input],
      _targets: Tensor[Target]
  ): NormalTensorDataset[Input, Target] = new NormalTensorDataset[Input, Target] {
    val features = _features
    val targets = _targets

    require(features.size.length > 0)
    require(features.size.head == targets.size.head)

    override def apply(i: Int): (Tensor[Input], Tensor[Target]) = (features(i), targets(i))

    override def getItem(idx: Int): (Tensor[Input], Tensor[Target]) = (features(idx), targets(idx))

    override def length: Int = features.size.head

    override def toString(): String =
      s"TensorDataset(features=${features.info}, targets=${targets.info})"
  }

}

//  def init(data: AnyRef*): Unit
//  def apply(data: AnyRef*): Unit =
//    init(data)
