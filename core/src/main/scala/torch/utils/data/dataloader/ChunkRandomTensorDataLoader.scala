package torch.utils.data.dataloader

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  DataLoaderOptions,
  ChunkMapTensorDataset,
  TensorExample,
  TensorExampleIterator,
  TensorExampleVectorIterator,
  ChunkSharedTensorBatchDataset,
  ExampleIterator,
  ExampleVectorOptional,
  FullDataLoaderOptions,
  ChunkRandomTensorDataLoader as CRTDL
}
import org.bytedeco.pytorch.DataLoaderOptions as DLOP
import torch.internal.NativeConverters.{fromNative, toNative}
import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}

object ChunkRandomTensorDataLoader {
  def apply(dataset: ChunkMapTensorDataset, option: TorchTensorDataLoaderOptions) =
    new ChunkRandomTensorDataLoader(
      dataset,
      option.batch_size,
      option.shuffle,
      option.num_workers,
      option.max_jobs,
      option.drop_last,
      option.in_order,
      option.timeout
    )
}

class ChunkRandomTensorDataLoader(
    dataset: ChunkMapTensorDataset,
    batch_size: Int,
    shuffle: Boolean = false,
    num_workers: Int = 0,
    max_jobs: Long = 0L,
    drop_last: Boolean = false,
    in_order: Boolean = true,
    timeout: Float = 0
) extends CRTDL(dataset, new DLOP())
    with TorchDataLoader
    with Iterable[TensorExample] {

  val option = TorchTensorDataLoaderOptions(
    batch_size = batch_size,
    shuffle = shuffle,
    num_workers = num_workers,
    max_jobs = max_jobs,
    drop_last = drop_last,
    in_order = in_order,
    timeout = timeout
  )

  private lazy val nativeDataLoader = new CRTDL(dataset, option.toNative)

  override def begin(): TensorExampleIterator = nativeDataLoader.begin()

  override def end(): TensorExampleIterator = nativeDataLoader.end()

  override def join(): Unit = nativeDataLoader.join()

  override def options(): FullDataLoaderOptions = nativeDataLoader.options()

  private val iteratorBuffer = new ListBuffer[TensorExample]()

  def getIteratorBuffer: mutable.Buffer[TensorExample] = {
    if (iteratorBuffer.length == 0) {
      val nativeDataLoader = new CRTDL(dataset, option.toNative)
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

  def iterator_raw: Iterator[TensorExample] = new Iterator[TensorExample] {

    private lazy val nativeDataLoader = new CRTDL(dataset, option.toNative)

    private var current: TensorExampleIterator =
      nativeDataLoader.begin()

    private val endIterator: TensorExampleIterator =
      nativeDataLoader.end()

    override def hasNext: Boolean = !current.equals(endIterator)

    override def next(): TensorExample = {
      val batch = current.access
      current = current.increment
      batch
    }
  }
}
