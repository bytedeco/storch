package torch
package utils
package trainer

import torch.nn.loss.CrossEntropyLoss
import org.bytedeco.javacpp.PointerScope
import scala.util.Using
import torch.*
//import torch.nn.modules.{HasParams, TensorModule}
//import torch.nn.loss.LossFunc
//import torch.optim.Optimizer
import torch.utils.data.{DataLoader}

/** Normal model Trainer for LSTM model LSTM模型训练器，继承自Trainable trait 使用AdamW优化器和CrossEntropyLoss损失函数
  */
class NormalTrainer[D <: BFloat16 | FloatNN: Default](
    model: LstmNet[D],
    train_loader: DataLoader[D],
    eval_loader: DataLoader[D],
    criterion: CrossEntropyLoss = new CrossEntropyLoss(),
    optimizer_type: String = "adamW",
    device: Device = Device.CPU
) extends Trainable[D, LstmNet[D]](
      model,
      train_loader,
      eval_loader,
      optimizer_type = "adamW",
      criterion = new CrossEntropyLoss(),
      device
    ) {

  /** * trian model
    * @param num_epochs
    * @param learning_rate
    * @param batch_size
    * @param eval_interval
    * @param feature_reshape
    * @return
    */
  override def train(
      num_epochs: Int,
      learning_rate: Double = 1e-3,
      batch_size: Int,
      eval_interval: Int = 200,
      feature_reshape: Seq[Int] = Seq(-1, 28, 28)
  ): LstmNet[D] = {
    // 创建AdamW优化器
    val optimizer = optimizer_type.toLowerCase match {
      case "adamw"   => torch.optim.AdamW(model.parameters, learning_rate)
      case "adam"    => torch.optim.Adam(model.parameters, learning_rate)
      case "sgd"     => torch.optim.SGD(model.parameters, learning_rate)
      case "rmsprop" => torch.optim.RMSprop(model.parameters, learning_rate)
      case "lbfgs"   => torch.optim.LBFGS(model.parameters, learning_rate)
      case "adagrad" => torch.optim.Adagrad(model.parameters, learning_rate)
      case _         => throw new IllegalArgumentException(s"不支持的优化器类型: $optimizer_type")
    }
//    val optimizer = AdamW(model.parameters(true), learning_rate)

    model.to(device)
//    val criterion = new CrossEntropyLoss()
    // use PointerScope to manage native resources
    Using.resource(new PointerScope()) { _ =>
      for (epoch <- 1 to num_epochs) {
        model.train()
        var totalLoss = 0.0f
        val exampleIter = train_loader.iterator
        var batchIndex = 0
        for ((inputs, targets) <- train_loader) {
          // move data to target device
          val inputsDevice = inputs.to(device)
          val targetsDevice = targets.to(device)
          val outputs = model(
            inputsDevice.reshape(feature_reshape*).to(dtype = implicitly[Default[D]].dtype)
          ) // maybe need reshape inputsDevice.reshape(-1, 28, 28).to(torch.float32)
          val loss = criterion(outputs, targetsDevice)
          // backpropagation and optimization
          optimizer.zeroGrad()
          loss.backward()
          optimizer.step()
          totalLoss = totalLoss + loss.item.asInstanceOf[Float]
          batchIndex = batchIndex + 1
          // evaluate model
          if (batchIndex % eval_interval == 0) {
            val (evalLoss, accuracy) = evaluate()
            println(
              s"Epoch: $epoch, Iteration: ${batchIndex}, | Training loss: ${loss.item}%.4f " +
                s"Train Loss: ${totalLoss / eval_interval}, Eval Loss: $evalLoss, Accuracy: $accuracy"
            )
            totalLoss = 0.0f
            model.train()
          }
        }

        // 每个epoch结束后评估
        val (evalLoss, accuracy) = evaluate()
        println(s"Epoch $epoch completed. Eval Loss: $evalLoss, Accuracy: $accuracy")
      }
    }

    model
  }

  /** * evaluate model
    * @param feature_reshape
    * @return
    *   (evalLoss, accuracy)
    */
  override def evaluate(feature_reshape: Seq[Int] = Seq(-1, 28, 28)): (Float, Float) = {
    model.eval()
    var totalLoss = 0.0f
    var correct = 0L
    var total = 0L
    // attention: must in  torch.noGrad{}
    torch.noGrad {
      for ((inputs, targets) <- eval_loader) {
        val inputsDevice = inputs.to(device)
        val targetsDevice = targets.to(device)
        val outputs = model(
          inputsDevice.reshape(feature_reshape*).to(dtype = implicitly[Default[D]].dtype)
        ) // maybe need reshape
        val loss = criterion(outputs, targetsDevice)
//        val rec: Float = loss.item
        totalLoss = totalLoss + loss.item.asInstanceOf[Float]
        // compute accuracy
        val predictions = outputs.argmax(dim = 1)
        correct += predictions.eq(targetsDevice).sum.item
        total += targetsDevice.numel
      }
    }

    val avgLoss = totalLoss / eval_loader.size
    val accuracy = correct.toFloat / total.toFloat
    (avgLoss, accuracy)
  }
}
