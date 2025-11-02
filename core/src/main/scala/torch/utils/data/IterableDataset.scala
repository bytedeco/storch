package torch
package utils
package data

import scala.collection.mutable.{Buffer, ListBuffer}

trait IterableDataset[
    Input <: BFloat16 | FloatNN: Default,
    Target <: BFloat16 | FloatNN | Int64: Default
](features: Tensor[Input], labels: Tensor[Target])
    extends Iterable[(Tensor[Input], Tensor[Target])] {

  require(features.size.length > 0)
  require(features.size.head == labels.size.head)

  private val iteratorBuffer = new ListBuffer[(Tensor[Input], Tensor[Target])]

  def length: Long = features.shape(0)

  def getItem(idx: Int): (Tensor[Input], Tensor[Target]) = (features(idx), labels(idx))

  def get_item(idx: Int): (Tensor[Input], Tensor[Target]) = getItem(idx)

  def getIteratorBuffer: Buffer[(Tensor[Input], Tensor[Target])] = {
    if (iteratorBuffer.length == 0) {
      (0 until length.toInt).foreach(idx => iteratorBuffer.append(getItem(idx)))
    }
    iteratorBuffer
  }

  override def iterator: Iterator[(Tensor[Input], Tensor[Target])] = {

    if (iteratorBuffer.length == 0) {
      getIteratorBuffer.iterator // only once ！ do not running twice
    } else {
      iteratorBuffer.iterator
    }

  }
}
