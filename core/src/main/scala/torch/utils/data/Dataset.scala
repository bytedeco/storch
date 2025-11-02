package torch
package utils
package data

import org.bytedeco.pytorch.{Example, ExampleVector, ExampleIterator}

trait Dataset[Input <: DType, Target <: DType] extends NormalDataset {

  def features: Tensor[Input]

  def targets: Tensor[Target]

  def length: Long

  def getItem(idx: Int): (Tensor[Input], Tensor[Target])

  def get_item(idx: Int): (Tensor[Input], Tensor[Target]) = getItem(idx)

  def get_batch(request: Long*): ExampleVector /// = super.get_batch(request*)

  def iterator: Iterator[(Tensor[Input], Tensor[Target])]

  def apply(idx: Int): (Tensor[Input], Tensor[Target]) = getItem(idx)

}

object Dataset {

  def apply[Input <: DType, Target <: DType](
      _features: Tensor[Input],
      _targets: Tensor[Target]
  ): Dataset[Input, Target] = new Dataset[Input, Target] {
    val features = _features
    val targets = _targets

    require(features.size.length > 0)
    require(features.size.head == targets.size.head)

    override def getItem(idx: Int): (Tensor[Input], Tensor[Target]) = (features(idx), targets(idx))

    override def apply(i: Int): (Tensor[Input], Tensor[Target]) = (features(i), targets(i))

    override def length: Long = features.size.head

//    override def size: Long = length

    override def toString(): String =
      s"TensorDataset(features=${features.info}, targets=${targets.info})"

    override def get_batch(request: Long*): ExampleVector = ??? // super.get_batch(request)

    override def iterator: Iterator[(Tensor[Input], Tensor[Target])] = ???
//      new Iterator[(Tensor[Input], Tensor[Target])] {
//        private var current: ExampleIterator = nativeDataLoader.begin
//        private val endIterator: ExampleIterator = nativeDataLoader.end
//
//        override def hasNext: Boolean = !current.equals(endIterator)
//
//        override def next(): (Tensor[Input], Tensor[Target]) = {
//          val batch = current.access
//          current = current.increment
//          exampleToTuple(batch)
//        }
//      }
  }
}

//with Iterable[(Tensor[? <: DType], Tensor[? <: DType])]

//  override def iterator: Iterator[(Tensor[? <: DType], Tensor[? <: DType])] =
//    new Iterator[(Tensor[? <: DType], Tensor[? <: DType])] {
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

//  override def size(): Int = super.size

//  def init(data: AnyRef*): Unit
//  def apply(data:AnyRef*):Unit =
//    init(data)
