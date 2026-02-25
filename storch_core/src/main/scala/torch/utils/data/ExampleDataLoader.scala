package torch.utils.data

import org.bytedeco.pytorch.*
import torch.utils.data.dataloader.sequential.SequentialDataLoader
import torch.utils.data.dataloader.TorchDataLoaderOptions
import torch.utils.data.datareader.ChunkDataReader
import org.bytedeco.javacpp.chrono.{Milliseconds, Seconds}
import torch.utils.data.sampler.{SequentialSampler, RandomSampler as TorchSampler}
import torch.{DType, Default}

import java.nio.file.Paths
import scala.collection.{Iterator, mutable}
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import torch.utils.data.Dataset as DatasetTrait
import torch.utils.data.dataset.normal.JavaDataset

class ExampleDataLoader[ParamType <: DType: Default](
    val dataset: DatasetTrait[ParamType, ? <: DType] | JavaDataset,
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
) extends Iterable[Example] {

  private val options = TorchDataLoaderOptions(
    batch_size = batch_size,
    shuffle = shuffle,
    sampler = sampler,
    batch_sampler = batch_sampler,
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

  private def createChunkDataReader(examples: Seq[Example]): ChunkDataReader = {
    val reader = new ChunkDataReader()
    val exampleVector = new org.bytedeco.pytorch.ExampleVector(examples*)
    reader(exampleVector)
    reader
  }

  private def createChunkDataset(
      reader: ChunkDataReader,
      examples: Seq[Example],
      options: TorchDataLoaderOptions
  ): ChunkDataset = {

    if (options.shuffle) {
      // for random sampler
      new ChunkDataset(
        reader,
        sampler,
        batch_sampler,
        new ChunkDatasetOptions(prefetch_factor.getOrElse(2), options.batch_size.toLong)
      )
    } else {
      // for sequential sampler
      new ChunkDataset(
        reader,
        sampler,
        batch_sampler,
        new ChunkDatasetOptions(prefetch_factor.getOrElse(2), options.batch_size.toLong)
      )
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
//    loaderOpts.timeout().put(new Milliseconds(options.timeout.toLong)) ////todo Javacpp Bug here timeout will make null pointer
    new ChunkRandomDataLoader(ds, loaderOpts)
  }

  private def createChunkSequentialDataLoader(
      ds: JavaDataset,
      sampler: SequentialSampler,
      options: TorchDataLoaderOptions
  ): SequentialDataLoader[ParamType] = {
    SequentialDataLoader[ParamType](ds, sampler, options)
  }

  private val examples = convertDatasetToExamples()
  private val reader = createChunkDataReader(examples)
  private val nativeDataset: ChunkDataset = createChunkDataset(reader, examples, options)
  private val sharedBatchDataset = createChunkSharedBatchDataset(nativeDataset)
  private lazy val nativeDataLoaderMain: ChunkRandomDataLoader =
    createChunkRandomDataLoader(sharedBatchDataset, options)

  private val iteratorBuffer = new ListBuffer[Example]()

  def getIteratorBuffer: mutable.Buffer[Example] = {
    if (iteratorBuffer.length == 0) {
      val nativeDataLoader: ChunkRandomDataLoader =
        createChunkRandomDataLoader(sharedBatchDataset, options)
      var current: ExampleIterator = nativeDataLoader.begin
      val endIterator: ExampleIterator = nativeDataLoader.end
      while (!current.equals(endIterator)) {
        val example = current.access
        iteratorBuffer.append(example)
        current = current.increment()
      }
    }
    iteratorBuffer
  }

  override def iterator: Iterator[Example] = {

    if (iteratorBuffer.length == 0) {
      getIteratorBuffer.iterator // only once ！ do not running twice
    } else {
      iteratorBuffer.iterator
    }
  }

  lazy val iteratorSeq: Seq[Example] = {

    if (iteratorBuffer.length == 0) {
      getIteratorBuffer.toSeq // only once ！ do not running twice
    } else {
      iteratorBuffer.toSeq
    }
  }

}

//override def iterator: Iterator[Example] = new Iterator[Example] {
//
//    private lazy val nativeDataLoader: ChunkRandomDataLoader =
//      createChunkRandomDataLoader(sharedBatchDataset, options)
//
//    private var current: ExampleIterator = nativeDataLoader.begin
//
//    private val endIterator: ExampleIterator = nativeDataLoader.end
//
//    override def hasNext: Boolean = !current.equals(endIterator)
//
//    override def next(): Example = {
//      val batch = current.access
//      current = current.increment
//      batch
//    }
//  }
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
