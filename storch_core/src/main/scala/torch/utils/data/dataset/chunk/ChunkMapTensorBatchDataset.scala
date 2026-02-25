package torch
package utils.data.dataset.chunk

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ChunkTensorDataReader,
  ExampleVectorOptional,
  SizeTArrayRef,
  SizeTOptional,
  TensorExampleVector,
  TensorMapper,
  TensorVector,
  ChunkBatchDataset as CBD,
  ChunkMapTensorBatchDataset as CMTBD,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.utils.data.dataset.chunk.ChunkMapTensorDataset

class ChunkMapTensorBatchDataset(chunkMapTensorDataset: ChunkMapTensorDataset)
    extends CMTBD(chunkMapTensorDataset) {

  override def get_batch(request: SizeTArrayRef): TensorExampleVector = super.get_batch(request)

  override def get_batch(request: Long*): TensorExampleVector = super.get_batch(request: _*)

  override def size(): SizeTOptional = super.size()
}

//  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)
