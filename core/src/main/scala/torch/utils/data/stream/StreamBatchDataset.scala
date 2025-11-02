package torch
package utils
package data
package stream

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{ExampleVector, SizeTOptional, JavaStreamBatchDataset as SBD}
import torch.utils.data.datareader.ExampleVectorReader

class StreamBatchDataset(reader: datareader.ExampleVectorReader) extends SBD(reader) {

  override def get_batch(request: Long): ExampleVector = super.get_batch(request)

  override def size(): SizeTOptional = super.size()
}
