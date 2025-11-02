package torch
package utils
package data

import org.bytedeco.javacpp.chrono.Milliseconds
import org.bytedeco.pytorch.*
import torch.utils.data.TensorDataset
import torch.utils.data.dataloader.{ChunkRandomTensorDataLoader, TorchTensorDataLoaderOptions}
import torch.utils.data.datareader.ChunkTensorDataReader
import torch.utils.data.dataset.normal.NormalTensorDataset
import torch.utils.data.sampler.RandomSampler as TorchSampler
import scala.collection.mutable
import java.nio.file.Paths
import scala.collection.Iterator
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import org.bytedeco.javacpp.chrono.{Milliseconds, Seconds}

class ExampleTensorDataLoader[ParamType <: DType: Default](
    val dataset: TensorDataset | NormalTensorDataset,
    val batch_size: Int,
    val shuffle: Boolean = false,
    val num_workers: Int = 0,
    val max_jobs: Long = 0L,
    val drop_last: Boolean = false,
    val in_order: Boolean = true,
    val sampler: TorchSampler,
    val batch_sampler: TorchSampler,
    val timeout: Float = 0,
    val pin_memory: Boolean = false,
    val prefetch_factor: Option[Int] = None,
    val persistent_workers: Boolean = false,
    val pin_memory_device: String = ""
) extends Iterable[TensorExample] {

  private val options = TorchTensorDataLoaderOptions(
    batch_size = batch_size,
    shuffle = shuffle,
    sampler = sampler,
    batch_sampler = batch_sampler,
    num_workers = num_workers,
    max_jobs = max_jobs,
    drop_last = drop_last,
    in_order = in_order,
    timeout = timeout,
    pin_memory = pin_memory,
    prefetch_factor = prefetch_factor,
    persistent_workers = persistent_workers
  )

  private def convertDatasetToTensorExamples(): Seq[TensorExample] = {
    val tensorExamples = new ArrayBuffer[TensorExample]()
    if (dataset.isInstanceOf[NormalTensorDataset]) {
      val normalDataset = dataset.asInstanceOf[NormalTensorDataset]
      for (i <- 0 until normalDataset.length.toInt) {
        val tensorExample = normalDataset.get(i)
        tensorExamples += tensorExample
      }
    } else {
      val datasetTrait = dataset.asInstanceOf[TensorDataset]
      for (i <- 0 until datasetTrait.length.toInt) {
        val data = datasetTrait.getItem(i)
        val tensorExample = new TensorExample(data.native)
        tensorExamples += tensorExample
      }
    }
    tensorExamples.toSeq
  }

  private def createTensorExampleVector(tensorExamples: Seq[TensorExample]): TensorExampleVector = {
    new TensorExampleVector(tensorExamples*)
  }

  private def createChunkTensorDataReader(
      tensorExampleVector: TensorExampleVector
  ): ChunkTensorDataReader = {
    val reader = new ChunkTensorDataReader()
    reader(tensorExampleVector)
    reader
  }

  private def createChunkTensorDataset(
      reader: ChunkTensorDataReader,
      tensorExamples: Seq[TensorExample],
      options: TorchTensorDataLoaderOptions
  ): ChunkTensorDataset = {
    if (options.shuffle) {
      // for random sampler
      new ChunkTensorDataset(
        reader,
        sampler,
        batch_sampler,
        new ChunkDatasetOptions(prefetch_factor.getOrElse(2), options.batch_size.toLong)
      )
    } else {
      // for sequential sampler
      new ChunkTensorDataset(
        reader,
        sampler,
        batch_sampler,
        new ChunkDatasetOptions(prefetch_factor.getOrElse(2), options.batch_size.toLong)
      )
    }
  }

  private def createChunkSharedTensorBatchDataset(
      chunkTensorDataset: ChunkTensorDataset
  ): ChunkMapTensorDataset = {
    new ChunkSharedTensorBatchDataset(chunkTensorDataset).map(new TensorExampleStack)
  }

  private def createChunkRandomTensorDataLoader(
      ds: org.bytedeco.pytorch.ChunkMapTensorDataset,
      options: TorchTensorDataLoaderOptions
  ): ChunkRandomTensorDataLoader = {
    val loaderOpts = new org.bytedeco.pytorch.DataLoaderOptions(options.batch_size)
    loaderOpts.batch_size.put(options.batch_size)
    loaderOpts.drop_last().put(options.drop_last)
    loaderOpts.enforce_ordering().put(options.in_order)
    loaderOpts.workers().put(options.num_workers)
    loaderOpts.max_jobs().put(options.max_jobs)
//    loaderOpts.timeout(new Milliseconds(new Seconds(options.timeout.toLong)))
//    loaderOpts.timeout().put(new Milliseconds(options.timeout.toLong)) ////todo Javacpp Bug here timeout will make null pointer
    ChunkRandomTensorDataLoader(ds, options)
  }

  private val tensorExamples = convertDatasetToTensorExamples()
  private val tensorExampleVector = createTensorExampleVector(tensorExamples)
  private val reader: ChunkTensorDataReader = createChunkTensorDataReader(tensorExampleVector)
  private val chunkTensorDataset: ChunkTensorDataset =
    createChunkTensorDataset(reader, tensorExamples, options)
  private val chunkSharedTensorBatchDataset: ChunkMapTensorDataset =
    createChunkSharedTensorBatchDataset(chunkTensorDataset)

  private lazy val nativeDataLoaderMain: ChunkRandomTensorDataLoader =
    createChunkRandomTensorDataLoader(chunkSharedTensorBatchDataset, options)

  private val iteratorBuffer = new ListBuffer[TensorExample]()

  def getIteratorBuffer: mutable.Buffer[TensorExample] = {
    if (iteratorBuffer.length == 0) {
      val nativeDataLoader: ChunkRandomTensorDataLoader =
        createChunkRandomTensorDataLoader(chunkSharedTensorBatchDataset, options)
      var current: TensorExampleIterator = nativeDataLoader.begin
      val endIterator: TensorExampleIterator = nativeDataLoader.end
      while (!current.equals(endIterator)) {
        val example = current.access
        iteratorBuffer.append(example)
        current = current.increment()
      }
    }
    iteratorBuffer
  }

  override def iterator: Iterator[TensorExample] = {

    if (iteratorBuffer.length == 0) {
      getIteratorBuffer.iterator // only once ！ do not running twice
    } else {
      iteratorBuffer.iterator
    }
  }

  lazy val iteratorSeq: Seq[TensorExample] = {

    if (iteratorBuffer.length == 0) {
      getIteratorBuffer.toSeq // only once ！ do not running twice
    } else {
      iteratorBuffer.toSeq
    }
  }

}

//  private val chunkMapTensorDataset = createChunkMapTensorDataset(chunkSharedTensorBatchDataset)
//    val prefetch_count = 1
//    new ChunkTensorDataset(
//      reader,
//      new RandomSampler(tensorExamples.size),
//      new RandomSampler(tensorExamples.size),
//      new org.bytedeco.pytorch.ChunkDatasetOptions(prefetch_count, options.batch_size)
//    )

//override def iterator: Iterator[TensorExample] = new Iterator[TensorExample] {
//
//  private lazy val nativeDataLoader: ChunkRandomTensorDataLoader =
//    createChunkRandomTensorDataLoader(chunkSharedTensorBatchDataset, options)
//
//  private var current: TensorExampleIterator =
//    nativeDataLoader.begin()
//  private val endIterator: TensorExampleIterator =
//    nativeDataLoader.end()
//
//  override def hasNext: Boolean = !current.equals(endIterator)
//
//  override def next(): TensorExample = {
//    val batch = current.access
//    current = current.increment
//    batch
//  }
//}
// 这里需要替换为实际的 ChunkRandomTensorDataLoader 构造函数
// 假设存在一个名为 ChunkRandomTensorDataLoader 的类
//    new org.bytedeco.pytorch.ChunkRandomTensorDataLoader(ds, loaderOpts)

//  private def createChunkMapTensorDataset(chunkSharedTensorBatchDataset: ChunkMapTensorBatchDataset): ChunkMapTensorDataset = {
//    chunkSharedTensorBatchDataset
//  }
