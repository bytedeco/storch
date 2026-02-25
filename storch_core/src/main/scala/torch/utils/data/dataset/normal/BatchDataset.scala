package torch.utils.data.dataset.normal

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{ExampleVector, SizeTArrayRef, SizeTOptional, JavaBatchDataset as BD}
import torch.utils.data.datareader
import torch.utils.data.datareader.ExampleVectorReader

class BatchDataset(reader: datareader.ExampleVectorReader) extends BD(reader) {

  override def size(): SizeTOptional = super.size()

  override def get_batch(request: SizeTArrayRef): ExampleVector = super.get_batch(request)

  override def get_batch(request: Long*): ExampleVector = super.get_batch(request: _*)
}
