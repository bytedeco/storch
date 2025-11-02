package torch.utils.data.dataset.custom

//package torch
//package utils
//package data
//package dataset
//
//import torch.Device.{CPU, CUDA}
//import torch.{Float32, Int64, Tensor}
//import torch.utils.data.dataset.Dataset
//import java.nio.file.Paths
//import scala.util.Random
//
//class MnistDataset extends Dataset[Float32] {
//
//  val device = if torch.cuda.isAvailable then CUDA else CPU
//
//  println(s"Using device: $device")
//  val dataPath = Paths.get("D:\\data\\FashionMNIST")
//  val train_dataset = FashionMNIST(dataPath, train = true, download = true)
//  val test_dataset = FashionMNIST(dataPath, train = false)
//  val trainFeatures = train_dataset.features.to(device)
//  val trainTargets = train_dataset.targets.to(device)
//  val r = Random(seed = 0)
//
//  def dataLoader: Iterator[(Tensor[Float32], Tensor[Int64])] =
//    r.shuffle(train_dataset).grouped(8).map { batch =>
//      val (features, targets) = batch.unzip
//      (torch.stack(features).to(device), torch.stack(targets).to(device))
//    }
//  val data = dataLoader
//  println(s"train_dataset.features.shape.head ${train_dataset.features.shape.head}")
//  override def length: Long = train_dataset.features.shape.head
//  override def getItem(idx: Int): (Tensor[Float32], Tensor[Int64]) = {
//    val feature: Tensor[Float32] = train_dataset.features(idx)
//    val target: Tensor[Int64] = train_dataset.targets(idx)
//    (feature,target)
//
//  }
//}
//
