package torch
package utils
package data

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.Tensor as TensorNative
import torch.utils.data.dataloader.{TorchDataLoader, TorchDataLoaderOptions}
import torch.utils.data.datareader.{ChunkDataReader, ExampleVectorReader}
import org.bytedeco.javacpp.chrono.{Milliseconds, Seconds}
import torch.utils.data.dataloader.sequential.SequentialDataLoader
import torch.utils.data.dataloader.random.RandomDataLoader
import torch.utils.data.sampler.{Sampler, SequentialSampler, RandomSampler}
import torch.{DType, Default, Tensor, *}

import scala.collection.mutable
import scala.collection.Iterator
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import torch.utils.data.Dataset as DatasetTrait
import torch.utils.data.dataset.normal.JavaDataset
import torch.internal.NativeConverters.{fromNative, toNative}

class DataVectorLoader[ParamType <: DType: Default](
    val dataset: DatasetTrait[ParamType, ? <: DType] | JavaDataset |
      NormalTensorDataset[ParamType, ? <: DType],
    val batch_size: Int,
    val shuffle: Boolean = false,
    val num_workers: Int = 0,
    val max_jobs: Long = 0L,
    val drop_last: Boolean = false,
    val in_order: Boolean = true,
    val sampler: Sampler | Option[Sampler] = None,
    val batch_sampler: Sampler | Option[Sampler] = None,
    val timeout: Float = 0.0f,
    val pin_memory: Boolean = false,
    val prefetch_factor: Option[Int] = None,
    val persistent_workers: Boolean = false,
    val pin_memory_device: String = ""
) extends Iterable[(Tensor[ParamType], Tensor[ParamType])] {

  val dataLength: Long = dataset match {
    case d: DatasetTrait[ParamType, ? <: DType]        => d.length
    case d: JavaDataset                                => d.length
    case d: NormalTensorDataset[ParamType, ? <: DType] => d.length
  }
  val chunkSampler = sampler match {
    case Some(sampler) => sampler
    case None          => new RandomSampler(dataLength)
    case s: Sampler    => s
  }

  val batchSampler = batch_sampler match {
    case Some(sampler) => sampler
    case None          => new RandomSampler(dataLength)
    case s: Sampler    => s
  }

  private val options = TorchDataLoaderOptions(
    batch_size = batch_size,
    shuffle = shuffle,
    sampler = chunkSampler,
    batch_sampler = batchSampler,
    num_workers = num_workers,
    max_jobs = max_jobs,
    pin_memory = pin_memory,
    drop_last = drop_last,
    in_order = in_order,
    timeout = timeout,
    prefetch_factor = prefetch_factor,
    persistent_workers = persistent_workers
  )

  private def convertDatasetToExamples(): Seq[Example] = {
    val examples = new ArrayBuffer[Example]()
    if (dataset.isInstanceOf[DatasetTrait[ParamType, ? <: DType]]) {
      val datasetTrait = dataset.asInstanceOf[DatasetTrait[ParamType, ? <: DType]]
      for (i <- 0 until datasetTrait.length.toInt) {
        val (data, target) = datasetTrait.getItem(i)
        val example = new Example(data.native, target.native)
        examples += example
      }
    } else if (dataset.isInstanceOf[NormalTensorDataset[ParamType, ? <: DType]]) {
      val normalTensorDataset = dataset.asInstanceOf[NormalTensorDataset[ParamType, ? <: DType]]
      for (i <- 0 until normalTensorDataset.length.toInt) {
        val (data, target) = normalTensorDataset.getItem(i)
        val example = new Example(data.native, target.native)
        examples += example
      }
    } else {
      val javaDataset = dataset.asInstanceOf[JavaDataset]
      for (i <- 0 until javaDataset.length.toInt) {
        val example = javaDataset.get(i)
        examples += example
      }
    }
    examples.toSeq
  }

  def exampleVectorToExample(exVec: ExampleVector): Example = {
    val example = new Example(exVec.get(0).data(), exVec.get(0).target())
    example
  }

  private def createExampleVectorReader(exampleSeq: Seq[Example]): ExampleVectorReader = {

    val reader = new ExampleVectorReader {
      override def exampleVec: ExampleVector = new org.bytedeco.pytorch.ExampleVector(exampleSeq*)
    }
//    reader(exampleSeq)
    reader
  }

  private def createJavaDataset(
      reader: ExampleVectorReader,
      examples: Seq[Example],
      options: TorchDataLoaderOptions
  ): JavaDataset = {

    if (options.shuffle) {
      // for random sampler
      new JavaDataset(reader)
    } else {
      // for sequential sampler
      new JavaDataset(reader)
    }

  }

  private def createChunkSharedBatchDataset(chunkDataset: ChunkDataset): ChunkMapDataset = {
    new ChunkSharedBatchDataset(chunkDataset).map(new ExampleStack)
  }

  private def createChunkRandomDataLoader(
      ds: ChunkMapDataset,
      options: TorchDataLoaderOptions
  ): ChunkRandomDataLoader = {
    val loaderOpts = new org.bytedeco.pytorch.DataLoaderOptions(options.batch_size)
    loaderOpts.batch_size.put(options.batch_size)
    loaderOpts.drop_last().put(options.drop_last)
    loaderOpts.enforce_ordering().put(options.in_order)
    loaderOpts.workers().put(options.num_workers)
    loaderOpts.max_jobs().put(options.max_jobs)
    //    loaderOpts.timeout(new Milliseconds(new Seconds(options.timeout.toLong)))
    //    loaderOpts.timeout.put(new Milliseconds(options.timeout.toLong)) //todo Javacpp Bug here timeout will make null pointer
    new ChunkRandomDataLoader(ds, loaderOpts)
  }

  private def createSequentialDataLoader(
      ds: JavaDataset,
//                                          sampler: SequentialSampler,
      options: TorchDataLoaderOptions
  ): SequentialDataLoader[ParamType] = {
    SequentialDataLoader[ParamType](ds, chunkSampler.asInstanceOf[SequentialSampler], options)
  }

  private def createRandomDataLoader(
      ds: JavaDataset,
      options: TorchDataLoaderOptions
  ): RandomDataLoader[ParamType] = {
    RandomDataLoader[ParamType](ds, chunkSampler.asInstanceOf[RandomSampler], options)
  }

  private val examples = convertDatasetToExamples()
  private val reader = createExampleVectorReader(examples)

  private val nativeDataset: JavaDataset = createJavaDataset(reader, examples, options)
//  private val sharedBatchDataset = createChunkSharedBatchDataset(nativeDataset)
//  private lazy val nativeDataLoaderMain: ChunkRandomDataLoader =
//    createChunkRandomDataLoader(sharedBatchDataset, options)

  private def exampleToTuple(example: Example): (Tensor[ParamType], Tensor[ParamType]) = {
    val feature = fromNative(example.data()).to(dtype = implicitly[Default[ParamType]].dtype)
    val target = fromNative(example.target()).to(dtype = implicitly[Default[ParamType]].dtype)
    (feature.reshape(feature.shape*), target.reshape(target.shape*))
  }

  private val iteratorBuffer = new ListBuffer[(Tensor[ParamType], Tensor[ParamType])]

  def getIteratorBuffer: mutable.Buffer[(Tensor[ParamType], Tensor[ParamType])] = {

    if (iteratorBuffer.length == 0) {
      if (options.shuffle || chunkSampler.isInstanceOf[RandomSampler]) {
        val nativeDataLoader = createRandomDataLoader(nativeDataset, options)
        var current = nativeDataLoader.begin()
        val endIterator: ExampleVectorIterator = nativeDataLoader.end()
        while (!current.equals(endIterator)) {
          val example = exampleVectorToExample(current.access)
          val feature =
            torch.from_native(example.data()).to(dtype = implicitly[Default[ParamType]].dtype)
          val target =
            torch.from_native(example.target()).to(dtype = implicitly[Default[ParamType]].dtype)
          iteratorBuffer.append(
            (feature.reshape(feature.shape*), target.reshape(target.shape*))
          )
          current = current.increment()
        }
      } else {
        val nativeDataLoader = createSequentialDataLoader(nativeDataset, options)
        var current = nativeDataLoader.begin()
        val endIterator: ExampleVectorIterator = nativeDataLoader.end()
        while (!current.equals(endIterator)) {
          val example = exampleVectorToExample(current.access)
          val feature =
            torch.from_native(example.data()).to(dtype = implicitly[Default[ParamType]].dtype)
          val target =
            torch.from_native(example.target()).to(dtype = implicitly[Default[ParamType]].dtype)
          iteratorBuffer.append(
            (feature.reshape(feature.shape*), target.reshape(target.shape*))
          )
          current = current.increment()
        }
      }

    }
    iteratorBuffer
  }

  override def iterator: Iterator[(Tensor[ParamType], Tensor[ParamType])] = {

    if (iteratorBuffer.length == 0) {
      getIteratorBuffer.iterator // only once ！ do not running twice
    } else {
      iteratorBuffer.iterator
    }

  }

  lazy val iteratorSeq: Seq[(Tensor[ParamType], Tensor[ParamType])] = {

    if (iteratorBuffer.length == 0) {
      getIteratorBuffer.toSeq // only once ！ do not running twice
    } else {
      iteratorBuffer.toSeq
    }
  }

}

//  override def iterator: Iterator[(Tensor[ParamType], Tensor[ParamType])] =
//    new Iterator[(Tensor[ParamType], Tensor[ParamType])] {
//
//      private var nativeDataLoader: ChunkRandomDataLoader =
//        createChunkRandomDataLoader(sharedBatchDataset, options)
//
//      private var current: ExampleIterator = nativeDataLoader.begin
//
//      private val endIterator: ExampleIterator = nativeDataLoader.end
//
//      override def hasNext: Boolean = !current.equals(endIterator)
//
//      override def next(): (Tensor[ParamType], Tensor[ParamType]) = {
//        val batch = current.access
//        current = current.increment
//        exampleToTuple(batch)
//      }
//    }

//    val prefetch_count = 1
//    new ChunkDataset(
//      reader,
//      new RandomSampler(examples.size),
//      new RandomSampler(examples.size),
//      new ChunkDatasetOptions(prefetch_count, options.batch_size.toLong)
//    )
//    dataset: Dataset[ParamType], batchSize: Int, shuffle: Boolean, sampler: Sampler, batchSampler: Sampler[List],
//    numWorkers: Int = 0, pinMemory: Boolean = false, dropLast: Boolean = false, timeout: Float = 0,
//    multiprocessingContext: ProcessContext = null, generator: Generator ,

//import torch.utils.data.dataloader.ChunkRandomDataLoader
//    train_loader = DataLoader(
//        train_ds,
//        batch_size=args.batch_size,
//        pin_memory=True,
//        drop_last=False,
//        shuffle=False,
//        num_workers=args.num_workers,
//        sampler=train_sampler
//    )
// dataset: Dataset[_T_co],
//        batch_size: Optional[int] = 1,
//        shuffle: Optional[bool] = None,
//        sampler: Union[Sampler, Iterable, None] = None,
//        batch_sampler: Union[Sampler[List], Iterable[List], None] = None,
//        num_workers: int = 0,
//        collate_fn: Optional[_collate_fn_t] = None,
//        pin_memory: bool = False,
//        drop_last: bool = False,
//        timeout: float = 0,
//        worker_init_fn: Optional[_worker_init_fn_t] = None,
//        multiprocessing_context=None,
//        generator=None,
//        *,
//        prefetch_factor: Optional[int] = None,
//        persistent_workers: bool = False,
//        pin_memory_device: str = "",
//        in_order: bool = True,
// 定义一个可迭代的类，用于遍历用户自定义数据集
// batchSize -> batch_size ,
// shuffle inOrder in_order-> enforce_ordering ,
// numWorkers -> num_workers ->workers,
// max_jobs
// pinMemory -> pin_memory ,
// dropLast -> drop_last ,
// timeout -> timeout ,
//
