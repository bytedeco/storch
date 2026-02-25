package torch
package utils
package trainer

//import torch.nn
import torch.*
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.loss.LossFunc
//import torch.nn.{Dropout, Embedding, Linear, Transformer, functional as F}
//import torch.::
//import torch.optim.Optimizer
import torch.utils.data.{Dataset, DataLoader}
import java.nio.file.Paths
import scala.math.*

trait Trainable[ParamType <: FloatNN: Default, M <: TensorModule[ParamType] with HasParams[
  ParamType
]](
    model: M,
    train_loader: DataLoader[ParamType],
    eval_loader: DataLoader[ParamType],
    optimizer_type: String = "adamW",
    criterion: LossFunc,
    device: Device = Device.CPU
):

  /** * train model
    * @param num_epochs
    * @param learning_rate
    * @param batch_size
    * @param eval_interval
    *   attention: you need to set Using.resource(new PointerScope()) { p =>
    * @return
    */
  def train(
      num_epochs: Int,
      learning_rate: Double = 1e-3,
      batch_size: Int,
      eval_interval: Int = 200,
      feature_reshape: Seq[Int]
  ): M

  /** * evaluate model performance
    * @return
    *   (loss, accuracy) attention: you need to set torch.noGrad() before call this method or in
    *   torch.noGrad() block or use torch.noGrad() { ... } block to ensure that model is in
    *   evaluation mode
    */
  def evaluate(feature_reshape: Seq[Int]): (Float, Float)
