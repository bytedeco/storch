//package torch
//package utils
//package trainer
//
//import torch.nn.loss.LossFunc
//import torch.nn
//import torch.*
//import torch.nn.modules.{HasParams, TensorModule}
//import torch.nn.{Dropout, Embedding, Linear, Transformer, functional as F}
//import torch.::
//import torch.optim.Optimizer
//import torch.nn.loss.CrossEntropyLoss
//import torch.utils.data.{Dataset, DataLoader}
//import java.nio.file.Paths
//import scala.math.*
//import java.nio.file.Paths
//import torch.internal.NativeConverters.{fromNative, toNative}
//import scala.util.{Random, Using}
//import org.bytedeco.javacpp.{FloatPointer, PointerScope}
//import torch.{BFloat16, Float32, Int64, Default, Tensor}
//import torch.nn.functional as F
//import torch.nn.modules.HasParams
//import torch.nn as nn
//import torch.optim.Adam
//import java.nio.file.Paths
//
//class LstmNets[D <: BFloat16 | Float32 : Default](
//                                                  inputSize: Int = 28,
//                                                  hiddenSize: Int = 128,
//                                                  numLayers: Int = 2,
//                                                  numClasses: Int = 10
//                                                ) extends HasParams[D] {
//
//  val lstm = register(nn.LSTM(inputSize, hiddenSize, numLayers, batch_first = true))
//  val fc = register(nn.Linear(hiddenSize, numClasses))
//
//  def apply(input: Tensor[D]): Tensor[D] = forward(input)
//
//  def forward(input: Tensor[D]): Tensor[D] =
//    val arr = Seq(numLayers, input.size.head, hiddenSize.toInt)
//    val h0 = torch.zeros(size = arr, dtype = input.dtype)
//    val c0 = torch.zeros(size = arr, dtype = input.dtype)
//    val outTuple3 = lstm(input, Some(h0), Some(c0))
//    var out: Tensor[D] = outTuple3._1
//    out = out.index(torch.indexing.::, -1, ::)
//    F.logSoftmax(fc(out), dim = 1)
//
//}
//class NormalTrainPipeline(
//                                                       model: LstmNet[Float32],
//                                                       train_loader: DataLoader[Float32],
//                                                       eval_loader: DataLoader[Float32],
//                                                       optimizer_type: String = "adamW",
//                                                       criterion: CrossEntropyLoss = nn.loss.CrossEntropyLoss(),
//                                                       device: Device = Device.CPU
//                                                       ) //extends Trainable[Float32, LstmNet[Float32]](model, train_loader, eval_loader, optimizer_type, criterion, device) {
//
//  def train(num_epochs: Int = 10, learning_rate: Float = 0.001f, batch_size: Int = 32, eval_interval: Int = 200): Unit =  {
////    model.train()
//    var total_loss = 0.0
//    val train_size = this.train_loader.size
//    val train_steps = train_size / batch_size
//    val optimizer = torch.optim.AdamW(this.model.parameters(true), lr = learning_rate)
//    (1 to num_epochs).foreach(epoch => {
//      var batchIndex = 0
//      val exampleIter = this.train_loader.iterator
//      val evalExampleIter = this.eval_loader.iterator
//      while (exampleIter.hasNext) {
//        Using.resource(new PointerScope()) { p =>
//          val batch = exampleIter.next
//          optimizer.zero_grad()
//          val feature = fromNative(batch.data).to(dtype = float32)
//          val target = fromNative(batch.target).to(dtype = float32)
//          val output = model.forward(feature)
//          val loss = criterion(
//            output.reshape(-1, output.size(-1)),
//            target.reshape(-1, target.size(-1))
//          )
//          loss.backward()
//          optimizer.step()
//          batchIndex += 1
//          total_loss += loss.item()
//          println(f"Epoch: $epoch | Batch: $batchIndex%4d | Training loss: ${loss.item}%.4f")
//          if batchIndex % 200 == 0 then
//            // run evaluation
//            torch.noGrad {
//              val correct = 0
//              val total = 0
//              if (evalExampleIter.hasNext){
//                val evalBatch = evalExampleIter.next
//                val evalBatchSize = evalBatch.data.size(0)
//                val evalFeatures = fromNative(evalBatch.data).to(dtype = float32)
//                val evalTargets = fromNative(evalBatch.target).to(dtype = float32)
//                val predictions = model.forward(evalFeatures.to(dtype = float32))
//                val evalLoss = criterion(predictions, evalTargets)
//                evalLoss.backward()
//                println(s"eval predictions : ${predictions} \n")
//                val accuracy = predictions.argmax(dim = 1).eq(evalTargets).sum / evalBatchSize
//                ).item
//                println(f"Epoch: $epoch | Batch: $batchIndex%4d | Training loss: ${loss.item}%.4f | Eval loss: ${evalLoss.item}%.4f | Eval accuracy: $accuracy%.4f")
//              }
//
//            }
//        }
//      }
//    }
//  }
//
//}
//
//
//
//
////class NormalTrainPipeline[ParamType <: FloatNN: Default, M <: TensorModule[ParamType] with HasParams[ParamType]] (
////                                                           model: M,
////                                                           train_loader: DataLoader[ParamType],
////                                                           optimizer: Optimizer,
////                                                           criterion: TensorModule[ParamType],
////                                                           device: Device = Device.CPU
////                                                                         ) extends Trainable[ParamType, M] {
////
////  override def train(
////                      model: M,
////                      train_loader: DataLoader[ParamType],
////                      optimizer: Optimizer,
////                      criterion: TensorModule[ParamType],
////                      num_epochs: Int,
////                      batch_size: Int,
////                      device: Device = Device.CPU
////  ): Double = {
////    model.train()
////    var total_loss = 0.0
////    val train_size = train_loader.size
////    val train_steps = train_size / batch_size
////
////    (1 to num_epochs).foreach(epoch => {
////      val exampleIter = train_loader.iterator
////      while (exampleIter.hasNext) {
////        val batch = exampleIter.next
////        // 前向传播
////        optimizer.zero_grad()
////        val output = model(batch)
////        // 计算损失
////        val loss = criterion(
////          output.reshape(-1, output.size(-1)),
////        )
////
////        // 反向传播
////        loss.backward()
////        optimizer.step()
////
////        total_loss += loss.item()
////      }
////      total_loss / train_loader.size
////    }
////  }
////
////
////}
