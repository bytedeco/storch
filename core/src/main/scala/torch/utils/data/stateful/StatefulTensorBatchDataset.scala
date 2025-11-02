package torch.utils.data.stateful

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  SizeTOptional,
  TensorExampleVectorOptional,
  JavaStatefulTensorBatchDataset as STBD
}
import torch.utils.data.datareader
import torch.utils.data.datareader.TensorExampleVectorReader

class StatefulTensorBatchDataset(reader: datareader.TensorExampleVectorReader)
    extends STBD(reader) {

  override def get_batch(request: Long): TensorExampleVectorOptional = super.get_batch(request)

  override def size(): SizeTOptional = super.size()
}
