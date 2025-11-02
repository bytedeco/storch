//package torch
//
///*
// * Copyright 2022 storch.dev
// *
// * Licensed under the Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// */
//
////> using scala "3.3"
////> using repository "sonatype:snapshots"
////> using repository "sonatype-s01:snapshots"
////> using lib "dev.storch::vision:0.0-2fff591-SNAPSHOT"
//// replace with pytorch-platform-gpu if you have a CUDA capable GPU
////> using lib "org.bytedeco:pytorch-platform:2.1.2-1.5.10"
//// enable for CUDA support
//////> using lib "org.bytedeco:cuda-platform-redist:12.3-8.9-1.5.10"
//// enable for native Apple Silicon support
//// will not be needed with newer versions of pytorch-platform
//////> using lib "org.bytedeco:pytorch:2.1.2-1.5.10,classifier=macosx-arm64"
//
//import org.bytedeco.javacpp.{FloatPointer, PointerScope}
//import org.bytedeco.pytorch.{OutputArchive, TensorExampleVectorIterator}
//import torch.{Float32, *}
//import torch.Device.{CPU, CUDA}
//import torch.utils.data.dataset.chunk.ChunkSharedBatchDataset
//import torch.nn.functional as F
//import torch.nn.modules.HasParams
//import torch.optim.Adam
////import torchvision.datasets.FashionMNIST
//
//import java.nio.file.Paths
////import scala.runtime.stdLibPatches.Predef.nn
//import scala.util.{Random, Using}
//import torch.internal.NativeConverters.{fromNative, toNative}
//
////// Define the model architecture
//class LstmNet[D <: BFloat16 | Float32: Default](
//    inputSize: Int = 28,
//    hiddenSize: Int = 128,
//    numLayers: Int = 2,
//    numClasses: Int = 10
//) extends HasParams[D] {
//
//  val lstm = register(nn.LSTM(inputSize, hiddenSize, numLayers, batch_first = true))
//  val fc = register(nn.Linear(hiddenSize, numClasses))
//
//  def apply(i: Tensor[D]): Tensor[D] =
//    val arr = Seq(numLayers, i.size.head, hiddenSize.toInt)
//    val h0 = torch.zeros(size = arr, dtype = i.dtype)
//    val c0 = torch.zeros(size = arr, dtype = i.dtype)
//    val outTuple3 = lstm(i, Some(h0), Some(c0))
//    var out: Tensor[D] = outTuple3._1
//    out = out.index(torch.indexing.::, -1, ::)
//    F.logSoftmax(fc(out), dim = 1)
//
//}
//
//class RnnNet[D <: BFloat16 | Float32: Default](
//    inputSize: Int = 28,
//    hiddenSize: Int = 128,
//    numLayers: Int = 2,
//    numClasses: Int = 10
//) extends HasParams[D] {
//
//  val rnn = register(nn.RNN(inputSize, hiddenSize, numLayers, batch_first = true))
//  val fc = register(nn.Linear(hiddenSize, numClasses))
//
//  def apply(i: Tensor[D]): Tensor[D] =
//    val arr = Seq(numLayers, i.size.head, hiddenSize.toInt)
//    val h0 = torch.zeros(size = arr, dtype = i.dtype)
//    val c0 = torch.zeros(size = arr, dtype = i.dtype)
//    val outTuple2 = rnn(i, Some(h0))
//    var out: Tensor[D] = outTuple2._1
//    out = out.index(torch.indexing.::, -1, ::)
//    F.logSoftmax(fc(out), dim = 1)
//
//}
//
//class GruNet[D <: BFloat16 | Float32: Default](
//    inputSize: Int = 28,
//    hiddenSize: Int = 128,
//    numLayers: Int = 2,
//    numClasses: Int = 10
//) extends HasParams[D] {
//
//  val gru = register(nn.GRU(inputSize, hiddenSize, numLayers, batch_first = true))
//  val fc = register(nn.Linear(hiddenSize, numClasses))
//
//  def apply(i: Tensor[D]): Tensor[D] =
//    val arr = Seq(numLayers, i.size.head, hiddenSize.toInt)
//    val h0 = torch.zeros(size = arr, dtype = i.dtype)
//    val c0 = torch.zeros(size = arr, dtype = i.dtype)
//    val outTuple2 = gru(i, Some(h0))
//    var out: Tensor[D] = outTuple2._1
//    out = out.index(torch.indexing.::, -1, ::)
//    F.logSoftmax(fc(out), dim = 1)
//
//}
//
///** Shows how to train a simple LstmNet on the MNIST dataset */
//object LstmNetApp extends App {
//
//  import org.bytedeco.javacpp.Loader
//  Loader.load(classOf[org.bytedeco.javacpp.presets.javacpp])
////  Loader.load(org.bytedeco.javacpp.presets.javacpp.class)
////  val model  = GruNet().to(device)
////  val model = LstmNet().to(device)
////  val model = RnnNet().to(device)
//  val model = LstmNet().to(CPU)
////  val model  = TransformerClassifier(embedding_dim = 128 , num_heads= 6, num_layers=6, hidden_dim = 30, num_classes=10, dropout_rate=0.1).to(device)
//  // prepare data FashionMNIST
//  //  val dataPath = Paths.get("data/mnist")
//  //  val mnistTrain = MNIST(dataPath, train = true, download = true)
//  //  val mnistEval = MNIST(dataPath, train = false)
//  // "D:\\code\\data\\FashionMNIST"
////  val dataPath = Paths.get("data/FashionMNIST")
//  import torch.FashionMNIST
//  val dataPath = Paths.get("D:\\data\\FashionMNIST")
//  val mnistTrain = FashionMNIST(dataPath, train = true, download = true)
//  val mnistEval = FashionMNIST(dataPath, train = false)
//  println(s"model ${model.modules.toSeq.mkString(" \n")}")
//  println(s"model ${model.summarize}")
//  val lossFn = torch.nn.loss.CrossEntropyLoss()
//  // enable AMSGrad to avoid convergence issues
//  val optimizer = Adam(model.parameters, lr = 1e-3, amsgrad = true)
//  val optimizerCopy = Adam(model.parameters, lr = 1e-3, amsgrad = true)
//  val device = if torch.cuda.isAvailable then CUDA else CPU
//  println(s"Using device: $device")
//  val evalFeatures = mnistEval.features.to(device)
//  val evalTargets = mnistEval.targets.to(device)
//  val r = Random(seed = 0)
//  def dataLoader: Iterator[(Tensor[Float32], Tensor[Int64])] =
//    r.shuffle(mnistTrain).grouped(8).map { batch =>
//      val (features, targets) = batch.unzip
//      (torch.stack(features).to(device), torch.stack(targets).to(device))
//    }
//
//  import org.bytedeco.pytorch.{
//    ChunkDataReader as CDR,
//    ChunkRandomDataLoader as CRDL,
//    ChunkDataset as CD,
//    ChunkSharedBatchDataset as CSBD,
//    ExampleIterator,
//    TensorExampleIterator,
//    JavaDistributedSequentialTensorDataLoader,
//    DistributedSequentialSampler as DSS,
//    DistributedRandomSampler as DRS,
//    JavaDistributedRandomTensorDataLoader,
//    JavaRandomTensorDataLoader,
//    SizeTArrayRef,
//    JavaSequentialTensorDataLoader,
//    JavaTensorDataset as TD,
//    TensorExampleStack,
//    JavaStreamDataset as JSD,
//    JavaStreamDataLoader,
//    StreamSampler as STS,
//    JavaRandomDataLoader,
//    ChunkDatasetOptions,
//    SizeTOptional,
//    JavaDataset,
//    ExampleVectorIterator,
//    RandomSampler,
//    DataLoaderOptions as DLO,
//    Example,
//    ExampleStack,
//    ExampleVector,
//    JavaStatefulDataset,
//    TensorExample,
//    TensorExampleVector,
//    AbstractTensor as Tensor
//  }
//  import torch.utils.data.datareader.{
//    ChunkDataReader,
//    ChunkTensorDataReader,
//    ExampleVectorReader,
//    TensorExampleVectorReader
//  }
//  import torch.utils.data.DataLoaderOptions
//  import torch.utils.data.sampler.{
//    DistributedRandomSampler,
//    DistributedSequentialSampler,
//    StreamSampler,
//    RandomSampler as RS,
//    SequentialSampler as SS
//  }
//  import torch.utils.data.dataloader.{
//    ChunkRandomDataLoader,
//    ChunkRandomTensorDataLoader,
//    DistributedRandomDataLoader,
//    DistributedRandomTensorDataLoader,
//    DistributedSequentialDataLoader,
//    DistributedSequentialTensorDataLoader,
//    RandomDataLoader,
//    RandomTensorDataLoader,
//    SequentialDataLoader,
//    SequentialTensorDataLoader,
//    StatefulDataLoader,
//    StatefulTensorDataLoader,
//    StreamDataLoader,
//    StreamTensorDataLoader
//  }
//  import torch.utils.data.dataset.{
//    ChunkDataset,
//    ChunkSharedBatchDataset,
//    ChunkSharedTensorBatchDataset,
//    ChunkStatefulDataset,
//    ChunkTensorDataset
//  }
//  import torch.utils.data.dataset.normal.{
//    JavaDataset as JD,
//    StatefulDataset,
//    StatefulTensorDataset,
//    StreamDataset,
//    StreamTensorDataset,
//    NormalTensorDataset
//  }
//
//  def exampleVectorToExample(exVec: ExampleVector): Example = {
//    val example = new Example(exVec.get(0).data(), exVec.get(0).target())
//    example
//  }
////  val exampleTensorSeq = mnistTrain.map(x => new TensorExample(x._1.native))
////  val tensorExampleVector = new TensorExampleVector(exampleTensorSeq*)
////  val reader = new ChunkTensorDataReader()// new TensorExampleVectorReader()
////  reader(tensorExampleVector)
//
//  val exampleSeq = mnistTrain.map(x => new Example(x._1.native, x._2.native))
////  val ex1 = new Example(mnistTrain.features.native ,mnistTrain.targets.native)
//  val exampleVector = new ExampleVector(exampleSeq*)
//  val reader = new ChunkDataReader()
////  {
////    override def read_chunk(chunk_index: Long) = exampleVector //  new ExampleVector(new Example(Tensor.create(10.0, 20.0, 50.0, 80.0, 100.0), Tensor.create(200.0)), new Example(Tensor.create(15.0, 30.0, 50.0, 80.0, 300.0), Tensor.create(400.0)), new Example(Tensor.create(20.0, 20.0, 50.0, 80.0, 100.0), Tensor.create(500.0)), new Example(Tensor.create(35.0, 30.0, 50.0, 80.0, 300.0), Tensor.create(600.0)), new Example(Tensor.create(40.0, 20.0, 50.0, 80.0, 100.0), Tensor.create(700.0)), new Example(Tensor.create(55.0, 30.0, 50.0, 80.0, 300.0), Tensor.create(800.0)), new Example(Tensor.create(60.0, 20.0, 50.0, 80.0, 100.0), Tensor.create(900.0)), new Example(Tensor.create(75.0, 30.0, 50.0, 80.0, 300.0), Tensor.create(300.0)))
////
////    override def chunk_count:Long = 1
////
////    override def reset(): Unit = {
////    }
////  }
//  reader(exampleVector)
////
////  val ds = new JavaDataset() {
////     val exampleVector = new ExampleVector(exampleSeq.toArray:_*)
////    override def get(index: Long): Example = exampleVector.get(index)
////
////    override def size = new SizeTOptional(exampleVector.size)
////
////  }
////  val ds = new JD(reader)//.map(new ExampleStack())
////val ds = new JSD() {
////  val exampleVector = reader.exampleVec
////
////  override def get_batch(size: Long): ExampleVector = exampleVector
////
////  override def size = new SizeTOptional(exampleVector.size)
////}
////val ds = new TD() {
////  val tex = reader.tensorExampleVec //new TensorExampleVector(new TensorExample(Tensor.create(10.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(15.0, 30.0, 50.0, 80.0, 300.0)), new TensorExample(Tensor.create(20.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(35.0, 30.0, 50.0, 80.0, 300.0)), new TensorExample(Tensor.create(40.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(55.0, 30.0, 50.0, 80.0, 300.0)), new TensorExample(Tensor.create(60.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(75.0, 30.0, 50.0, 80.0, 300.0)))
////
////  override def get(index: Long): TensorExample = {
////    tex.get(index)
////    //                    return super.get(index);
////  }
////
////  override def get_batch(indices: SizeTArrayRef): TensorExampleVector = tex //.get_batch(indices) // ds.get_batch(indices) // exampleVector
////
////  override def size = new SizeTOptional(tex.size)
////}
//  val batch_size = 32
//  val prefetch_count = 1
////  val ds = new ChunkSharedBatchDataset(new ChunkDataset(reader, new RandomSampler(exampleSeq.size), new RandomSampler(exampleSeq.size), new ChunkDatasetOptions(prefetch_count, batch_size))).map(new ExampleStack)
//  //  val ds  = new ChunkSharedTensorBatchDataset(new ChunkTensorDataset(reader,new RS(exampleTensorSeq.size),new ChunkDatasetOptions(prefetch_count, batch_size))).map(new TensorExampleStack)
//  val ds = new ChunkSharedBatchDataset(
//    new ChunkDataset(
//      reader,
//      new RandomSampler(exampleSeq.size),
//      new RandomSampler(exampleSeq.size),
//      new ChunkDatasetOptions(prefetch_count, batch_size)
//    )
//  ).map(new ExampleStack)
//
//  //  val ds = new TensorDataset(reader)
////  val ds = new StreamDataset(reader)
//  val opts = new DataLoaderOptions(32)
////  opts.workers.put(5)
//  opts.batch_size.put(32)
////  opts.enforce_ordering.put(true)
////  opts.drop_last.put(false)
//  val data_loader = new ChunkRandomDataLoader(ds, opts)
////  val data_loader = new ChunkRandomTensorDataLoader(ds, opts)
////  val data_loader = new JavaDistributedSequentialTensorDataLoader(ds, new DSS(ds.size.get), opts)
////  val data_loader = new JavaDistributedRandomTensorDataLoader(ds, new DRS(ds.size.get), opts)
////  val data_loader = new JavaSequentialTensorDataLoader(ds, new SS(ds.size.get), opts)
////  val data_loader = new JavaStreamDataLoader(ds, new STS(ds.size.get), opts)
////  val data_loader = new JavaStreamDataLoader(ds, new STS(ds.size.get), opts)
////  val data_loader = new JavaStreamDataLoader(ds, new StreamSampler(ds.size.get), opts)
////  val data_loader = new RandomDataLoader(ds, new RS(ds.size.get), opts)
////  val data_loader = new SequentialDataLoader(ds, new SS(ds.size.get), opts)
////  val data_loader = new DistributedSequentialDataLoader(ds, new DistributedSequentialSampler(ds.size.get), opts)
////  val data_loader = new DistributedRandomDataLoader(ds, new DistributedRandomSampler(ds.size.get), opts)
//  //  val data_loader = new JavaRandomDataLoader(ds, new RandomSampler(ds.size.get), opts)
//  println(s"ds.size.get {ds.size.get} data_loader option ${data_loader.options.batch_size()}")
//  for (epoch <- 1 to 2) {
////    var it: ExampleVectorIterator = data_loader.begin
////    var it :TensorExampleVectorIterator = data_loader.begin
////    var it :TensorExampleIterator = data_loader.begin
//    var it: ExampleIterator = data_loader.begin
//    var batchIndex = 0
//    println("coming in for loop")
//    while (!it.equals(data_loader.end)) {
//      Using.resource(new PointerScope()) { p =>
//        val batch = it.access
//        optimizer.zeroGrad()
//        val trainDataTensor = fromNative(batch.data())
//        val prediction = model(fromNative(batch.data()).reshape(-1, 28, 28))
//        val loss = lossFn(prediction, fromNative(batch.target()))
//        loss.backward()
//        optimizer.step()
//        it = it.increment
//        batchIndex += 1
//        if batchIndex % 200 == 0 then
//          // run evaluation
//          val predictions = model(evalFeatures.reshape(-1, 28, 28))
//          val evalLoss = lossFn(predictions, evalTargets)
//          val featuresData = new Array[Float](1000)
//          val fp4 = new FloatPointer(predictions.native.data_ptr_float())
//          fp4.get(featuresData)
//          println(s"\n ffff size ${featuresData.size} shape ${evalFeatures.shape
//              .mkString(", ")}a data ${featuresData.mkString(" ")}")
//          println(s"predictions : ${predictions} \n")
//          val accuracy =
//            (predictions.argmax(dim = 1).eq(evalTargets).sum / mnistEval.length).item
//          println(
//            f"Epoch: $epoch | Batch: $batchIndex%4d | Training loss: ${loss.item}%.4f | Eval loss: ${evalLoss.item}%.4f | Eval accuracy: $accuracy%.4f"
//          )
////        it = it.increment
//
//      }
//    }
//    optimizerCopy.add_parameters(model.namedParameters()) //
//    println(s"optimizerCopy ${optimizerCopy}")
//    println(s"optimizer ${optimizer}")
//    println(s"judge optimizer ${optimizer == optimizerCopy}")
//    println(s"model parameters dict ${model.namedParameters()}")
//  }
//
//  // a.index(indexArrayRefA).add_(torch.mul(x.index(indexArrayRefX), a_prev.index(indexArrayRefA_prev)))
//  class PositionalEncoding[D <: BFloat16 | Float32: Default](d_model: Long, max_len: Long = 28 * 28)
//      extends HasParams[D] {
//
//    import torch.{---, Slice}
//
//    var tensor = torch.ones(Seq(4, 4))
//    println(s"First row: ${tensor(0)}")
//    // First row: tensor dtype=float32, shape=[4], device=CPU
//    // [1.0000, 1.0000, 1.0000, 1.0000]
//    println(s"First column: ${tensor(Slice(), 0)}")
//    // First column: tensor dtype=float32, shape=[4], device=CPU
//    // [1.0000, 1.0000, 1.0000, 1.0000]
//    println(s"Last column: ${tensor(---, -1)}")
//
//    val arr = Seq(max_len, d_model)
//    var encoding = torch.zeros(size = arr.map(_.toInt), dtype = this.paramType)
//    val position = torch.arange(0, max_len, dtype = this.paramType).unsqueeze(1)
//    val div_term =
//      torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.Tensor(10000.0)) / d_model))
//    val sinPosition = torch.sin(position * div_term).to(dtype = this.paramType)
//    val cosPosition = torch.cos(position * div_term).to(dtype = this.paramType)
//    val indexSin = torch.Tensor(Seq(0L, 1L))
//    val indexCos = torch.Tensor(Seq(1L, 1L))
//    encoding.index(::, 1.::(13)).add(sinPosition)
//    encoding.index(::, Seq[Long](2, 1, 13)).add(sinPosition)
//    encoding.index(::, 13).equal(sinPosition)
//    encoding.update(indices = Seq(2.::(21), 1.::(13)), values = sinPosition)
//    encoding.update(indices = Seq(---, 2.::(21), 1.::(13)), values = sinPosition)
//    encoding.update(indices = Seq(---, ::(21), 1.::(13)), values = sinPosition)
//    encoding.update(indices = Seq(---, 1.::, 1.::(13)), values = sinPosition)
//    encoding.update(indices = Seq(---, ::, 1.::(13)), values = sinPosition)
//    encoding = encoding.to(dtype = this.paramType)
//    encoding = torch.indexCopy(encoding, 0, indexSin, sinPosition)
//    encoding = torch.indexCopy(encoding, 0, indexCos, cosPosition)
//    encoding = encoding.unsqueeze(0)
//
//    // return x + self.encoding[: ,: x.size(1)].to(x.device)
//    def apply(x: torch.Tensor[D]): torch.Tensor[D] =
//      x.add(encoding).to(x.device)
//  }
//
//  // run training
////  for (epoch <- 1 to 50) do
////    for (batch <- dataLoader.zipWithIndex) do
////      // make sure we deallocate intermediate tensors in time shape [32,1,28,28]
////      Using.resource(new PointerScope()) { p =>
////        val ((feature, target), batchIndex) = batch
////        optimizer.zeroGrad()
////        val prediction = model(feature.reshape(-1,28,28))
////        val loss = lossFn(prediction, target)
////        loss.backward()
////        optimizer.step()
////        if batchIndex % 200 == 0 then
////          // run evaluation
////          val predictions= model(evalFeatures.reshape(-1,28,28))
////          val evalLoss = lossFn(predictions, evalTargets)
////          val featuresData  = new Array[Float](1000)
////          val fp4 = new FloatPointer(predictions.native.data_ptr_float())
////          fp4.get(featuresData)
////          println(s"\n ffff size ${featuresData.size} shape ${evalFeatures.shape.mkString(", ")}a data ${featuresData.mkString(" " )}")
////          println(s"predictions : ${predictions} \n")
////          val accuracy =
////            (predictions.argmax(dim = 1).eq(evalTargets).sum / mnistEval.length).item
////          println(
////            f"Epoch: $epoch | Batch: $batchIndex%4d | Training loss: ${loss.item}%.4f | Eval loss: ${evalLoss.item}%.4f | Eval accuracy: $accuracy%.4f"
////          )
////      }
////    val archive = new OutputArchive
////    model.save(archive)
////    archive.save_to("net.pt")
//}
