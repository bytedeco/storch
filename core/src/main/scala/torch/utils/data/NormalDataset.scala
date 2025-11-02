package torch
package utils
package data

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  Example,
  ExampleVector,
  SizeTArrayRef,
  SizeTOptional,
  JavaDataset as SD
}

trait NormalDataset extends SD {

  override def size(): SizeTOptional = super.size()

  override def position(position: Long): SD = super.position(position)

  override def getPointer(i: Long): SD = super.getPointer(i)

  override def get(index: Long): Example = super.get(index)

//  override def get_batch(indices: SizeTArrayRef): ExampleVector = super.get_batch(indices)

//  def get_batch(request: Seq[Long]): ExampleVector = super.get_batch(request*)
}
