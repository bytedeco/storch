//package torch
//package utils
//package trainer
//
//import torch.nn
//import torch.*
//import torch.nn.modules.{HasParams, TensorModule}
//import torch.nn.loss.LossFunc
//import torch.optim.Optimizer
//import torch.utils.data.{Dataset, DataLoader}
//
//class SimpleTrainable[ParamType <: FloatNN: Default, M <: TensorModule[ParamType] with HasParams[ParamType]] private (
//    model: M,
//    train_loader: DataLoader[ParamType],
//    eval_loader: DataLoader[ParamType],
//    optimizer_type: String = "adamW",
//    criterion: LossFunc,
//    device: Device = Device.CPU
//) extends Trainable[ParamType, M](model, train_loader, eval_loader, optimizer_type, criterion, device) {
//
//  // 实现train方法
//  override def train(num_epochs: Int, learning_rate: Float = 0.001f, batch_size: Int, eval_interval: Int = 200): M = {
//    // 将模型移动到指定设备
//    model.to(device)
//
//    // 创建优化器
//    val optimizer = optimizer_type.toLowerCase match {
//      case "adamw" => torch.optim.AdamW(model.parameters, learning_rate)
//      case "adam" => torch.optim.Adam(model.parameters, learning_rate)
//      case "sgd" => torch.optim.SGD(model.parameters, learning_rate)
//      // 可以根据需要添加更多优化器类型
//      case _ => throw new IllegalArgumentException(s"不支持的优化器类型: $optimizer_type")
//    }
//
//    // 训练循环
//    for (epoch <- 1 to num_epochs) {
//      model.train()
//      var running_loss = 0.0f
//      var iteration = 0
//
//      // 遍历训练数据
//      for (batch <- train_loader) {
//        iteration += 1
//        val (inputs, targets) = batch
//
//        // 将输入和目标移动到设备
//        val inputsOnDevice = inputs.to(device)
//        val targetsOnDevice = targets.to(device)
//
//        // 前向传播
//        val outputs = model(inputsOnDevice)
//        val loss = criterion(outputs, targetsOnDevice)
//
//        // 反向传播和优化
//        optimizer.zero_grad()
//        loss.backward()
//        optimizer.step()
//
//        running_loss += loss.item()
//
//        // 定期评估
//        if (iteration % eval_interval == 0) {
//          val avg_loss = running_loss / eval_interval
//          println(s"Epoch $epoch, Iteration $iteration, Loss: $avg_loss")
//          running_loss = 0.0f
//
//          // 进行评估
//          evaluate()
//        }
//      }
//
//      // 每个epoch结束后进行一次完整评估
//      model.eval()
//      println(s"Epoch $epoch 训练完成，开始评估...")
//      evaluate()
//    }
//
//    // 返回训练好的模型
//    model
//  }
//
//  // 辅助方法：评估模型性能
//  private def evaluate(): (Float, Float) = {
//    model.eval()
//    var total_loss = 0.0f
//    var correct = 0
//    var total = 0
//
//    torch.noGrad() {
//      for (batch <- eval_loader) {
//        val (inputs, targets) = batch
//        val inputsOnDevice = inputs.to(device)
//        val targetsOnDevice = targets.to(device)
//
//        val outputs = model(inputsOnDevice)
//        val loss = criterion(outputs, targetsOnDevice)
//
//        total_loss += loss.item()
//
//        // 假设是分类任务，计算准确率
//        // 这里可能需要根据具体任务进行调整
//        if (outputs.dim == 2) {
//          val predicted = outputs.argmax(dim = 1)
//          total += targetsOnDevice.size(0)
//          correct += predicted.eq(targetsOnDevice).sum().item().toInt
//        }
//      }
//    }
//
//    val avg_loss = total_loss / eval_loader.length
//    val accuracy = if (total > 0) (correct.toFloat / total) * 100 else 0.0f
//    println(s"评估结果 - Loss: $avg_loss, Accuracy: $accuracy%")
//
//    // 恢复训练模式
//    model.train()
//  }
//}
//
//// 伴生对象，提供工厂方法
//object SimpleTrainable {
//  def apply[ParamType <: FloatNN: Default, M <: TensorModule[ParamType] with HasParams[ParamType]]
//      (model: M,
//       train_loader: DataLoader[ParamType],
//       eval_loader: DataLoader[ParamType],
//       optimizer_type: String = "adamW",
//       criterion: LossFunc,
//       device: Device = Device.CPU
//      ): SimpleTrainable[ParamType, M] = {
//    new SimpleTrainable(model, train_loader, eval_loader, optimizer_type, criterion, device)
//  }
//}
