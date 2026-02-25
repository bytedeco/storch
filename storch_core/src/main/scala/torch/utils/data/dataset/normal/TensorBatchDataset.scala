package torch.utils.data.dataset.normal

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  SizeTArrayRef,
  SizeTOptional,
  TensorExampleVector,
  JavaTensorBatchDataset as TBD
}
import torch.utils.data.datareader
import torch.utils.data.datareader.TensorExampleVectorReader

class TensorBatchDataset(reader: datareader.TensorExampleVectorReader) extends TBD(reader) {

  override def size(): SizeTOptional = super.size()

  override def get_batch(request: SizeTArrayRef): TensorExampleVector = super.get_batch(request)

  override def get_batch(request: Long*): TensorExampleVector = super.get_batch(request: _*)
}
