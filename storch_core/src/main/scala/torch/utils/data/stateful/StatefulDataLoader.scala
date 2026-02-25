package torch
package utils
package data
package stateful

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ExampleVector,
  ExampleVectorIterator,
  FullDataLoaderOptions,
  JavaStatefulDataLoader as SDL
}
import org.bytedeco.pytorch.DataLoaderOptions as DLOP
import torch.utils.data.dataloader.{TorchDataLoader, TorchDataLoaderOptions}
import torch.utils.data.dataloader.TorchDataLoaderOptions
import scala.collection.mutable
import scala.collection.mutable.{ListBuffer}

object StatefulDataLoader {

  def apply(dataset: StatefulDataset, option: TorchDataLoaderOptions) =
    new StatefulDataLoader(
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

class StatefulDataLoader(
    dataset: StatefulDataset,
    batch_size: Int,
    shuffle: Boolean = false,
    num_workers: Int = 0,
    max_jobs: Long = 0L,
    drop_last: Boolean = false,
    in_order: Boolean = true,
    timeout: Float = 0
) extends SDL(dataset, new DLOP())
    with TorchDataLoader
    with Iterable[ExampleVector] {

  val option = TorchDataLoaderOptions(
    batch_size = batch_size,
    shuffle = shuffle,
    num_workers = num_workers,
    max_jobs = max_jobs,
    drop_last = drop_last,
    in_order = in_order,
    timeout = timeout
  )

  private lazy val nativeDataLoader = new SDL(dataset, option.toNative)

  override def begin(): ExampleVectorIterator = nativeDataLoader.begin()

  override def end(): ExampleVectorIterator = nativeDataLoader.end()

  override def join(): Unit = nativeDataLoader.join()

  override def options(): FullDataLoaderOptions = nativeDataLoader.options()

  private val iteratorBuffer = new ListBuffer[ExampleVector]()

  def getIteratorBuffer: mutable.Buffer[ExampleVector] = {
    if (iteratorBuffer.length == 0) {
      val nativeDataLoader = new SDL(dataset, option.toNative)
      var current: ExampleVectorIterator = nativeDataLoader.begin
      val endIterator: ExampleVectorIterator = nativeDataLoader.end
      while (!current.equals(endIterator)) {
        val example = current.access
        iteratorBuffer.append(example)
        current = current.increment()
      }
    }
    iteratorBuffer
  }

  override def iterator: Iterator[ExampleVector] = {
    if (iteratorBuffer.length == 0) {
      getIteratorBuffer.iterator // only once ！ do not running twice
    } else {
      iteratorBuffer.iterator
    }
  }

  lazy val iteratorSeq: Seq[ExampleVector] = {
    if (iteratorBuffer.length == 0) {
      getIteratorBuffer.toSeq // only once ！ do not running twice
    } else {
      iteratorBuffer.toSeq
    }
  }

  def iterator_raw: Iterator[ExampleVector] = new Iterator[ExampleVector] {

    private lazy val nativeDataLoader = new SDL(dataset, option.toNative)

    private var current: ExampleVectorIterator = nativeDataLoader.begin()

    private val endIterator: ExampleVectorIterator = nativeDataLoader.end()

    override def hasNext: Boolean = !current.equals(endIterator)

    override def next(): ExampleVector = {
      val batch = current.access
      current = current.increment
      batch
    }
  }
}
