package torch.utils.data.dataloader

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  DataLoaderOptions,
  ChunkMapDataset,
  ExampleIterator,
  ExampleStack,
  ExampleVectorOptional,
  FullDataLoaderOptions,
  Example,
  ChunkBatchDataset as CBD,
  ChunkRandomDataLoader as CRDL
}
import org.bytedeco.pytorch.DataLoaderOptions as DLOP
import torch.internal.NativeConverters.{fromNative, toNative}
import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}

object ChunkRandomDataLoader {
  def apply(dataset: ChunkMapDataset, option: TorchDataLoaderOptions) =
    new ChunkRandomDataLoader(
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

class ChunkRandomDataLoader(
    dataset: ChunkMapDataset,
    batch_size: Int,
    shuffle: Boolean = false,
    num_workers: Int = 0,
    max_jobs: Long = 0L,
    drop_last: Boolean = false,
    in_order: Boolean = true,
    timeout: Float = 0
) extends CRDL(dataset, new DLOP())
    with TorchDataLoader
    with Iterable[Example] {

  val option = TorchDataLoaderOptions(
    batch_size = batch_size,
    shuffle = shuffle,
    num_workers = num_workers,
    max_jobs = max_jobs,
    drop_last = drop_last,
    in_order = in_order,
    timeout = timeout
  )

  private lazy val nativeDataLoader = new CRDL(dataset, option.toNative)

  override def begin(): ExampleIterator = nativeDataLoader.begin()

  override def end(): ExampleIterator = nativeDataLoader.end()

  override def join(): Unit = nativeDataLoader.join()

  override def options(): FullDataLoaderOptions = new FullDataLoaderOptions(option.toNative)

  private val iteratorBuffer = new ListBuffer[Example]()

  def getIteratorBuffer: mutable.Buffer[Example] = {

    if (iteratorBuffer.length == 0) {
      val nativeDataLoader = new CRDL(dataset, option.toNative)
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

  def iterator_raw: Iterator[Example] = new Iterator[Example] {

    private lazy val nativeDataLoader = new CRDL(dataset, option.toNative)

    private var current: ExampleIterator = nativeDataLoader.begin()

    private val endIterator: ExampleIterator = nativeDataLoader.end()

    override def hasNext: Boolean = !current.equals(endIterator)

    override def next(): Example = {
      val batch = current.access
      current = current.increment
      batch
    }
  }
}
