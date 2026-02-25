package torch
package utils
package data
package stream

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{ExampleVector, SizeTOptional, JavaStreamDataset as JSD}
import torch.utils.data.datareader.ExampleVectorReader

class StreamDataset(reader: datareader.ExampleVectorReader) extends JSD {

  val ds = new JSD() {
    val exampleVector = reader.exampleVec

    override def get_batch(size: Long): ExampleVector = exampleVector

    override def size = new SizeTOptional(exampleVector.size)
  }
  override def get_batch(request: Long): ExampleVector = ds.get_batch(request) // reader.exampleVec

  override def size(): SizeTOptional = new SizeTOptional(reader.exampleVec.size)

}

//  override def position(position: Long): SD = super.position(position)
