package torch
package utils
package trainer

//import org.bytedeco.pytorch
//import org.bytedeco.javacpp.{FloatPointer, PointerScope}
//import org.bytedeco.pytorch.{Example, InputArchive, OutputArchive, TensorExampleVectorIterator}
//import org.bytedeco.pytorch.{
//  ChunkDatasetOptions,
//  Example,
//  ExampleIterator,
//  ExampleStack,
//  ExampleVector
//}
//import org.bytedeco.pytorch.global.torch as torchNative
//import java.net.URL
//import java.util.zip.GZIPInputStream
import java.nio.file.{Paths, Files, Path}
//import scala.collection.{mutable, Set as KeySet}
//import scala.util.{Try, Success, Failure, Using, Random}
import torch.Device.{CPU, CUDA}
//import torch.internal.NativeConverters.{fromNative, toNative}
//import torch.{---, ::, &&, Slice}
//import torch.nn.functional as F
//import torch.nn.modules.{HasParams, TensorModule}
import torch.optim.Adam
//import torch.{BFloat16, Float32, Int64, Default, Tensor}
//import torch.nn as nn
//import torch.utils.data.{NormalTensorDataset, DataLoaderOptions, Dataset, DataLoader}
import torch.utils.data.dataset.custom.{FashionMNIST}
import torch.utils.data.*
//import torch.utils.data.dataloader.*
//import torch.utils.data.datareader.ChunkDataReader
import torch.utils.data.dataset.*
import torch.utils.data.sampler.RandomSampler
//import torch.numpy.TorchNumpy
//import torch.numpy.enums.DType.Float32 as NPFloat32
//import torch.numpy.matrix.NDArray
//import TorchNumpy.loadNDArrayFromCSV
//import torch.pandas.DataFrame

object LstmTrainerExample {
  def main(args: Array[String]): Unit = {
    // defined model parameters
    val inputSize = 28
    val hiddenSize = 128
    val numLayers = 2
    val numClasses = 10
    val batchSize = 1000
    val numEpochs = 10
    val learningRate = 1e-3
    val timeout = 10.0f
    val modelPahth = "./data/lstm-net.pt"
    val dataPath = Paths.get("./data//FashionMNIST")
    val device = if torch.cuda.isAvailable then CUDA else CPU

    val optimizer_type: String = "adamW"
    val mnistTrain = FashionMNIST(dataPath, train = true, download = true)
    val mnistEval = FashionMNIST(dataPath, train = false)
    val sampler = new RandomSampler(mnistTrain.length)
    val batchSampler = new RandomSampler(mnistTrain.length)
    val criterion = torch.nn.loss.CrossEntropyLoss()
    val model = new LstmNet[Float32](inputSize, hiddenSize, numLayers, numClasses).to(device)
    val optimizer = Adam(model.parameters, lr = learningRate, amsgrad = true)

    val trainLoader = new DataLoader(
      dataset = mnistTrain,
      batch_size = batchSize,
      shuffle = true,
      sampler = sampler,
      batch_sampler = batchSampler,
      timeout = timeout
    )
    val evalLoader = new DataLoader(
      dataset = mnistEval,
      batch_size = batchSize,
      shuffle = false,
      sampler = sampler,
      batch_sampler = batchSampler,
      timeout = timeout
    )
    val trainer =
      new NormalTrainer(model, trainLoader, evalLoader, criterion, optimizer_type, device)
    trainer.train(numEpochs, learningRate, batchSize, feature_reshape = Seq(-1, 28, 28))
  }
}
