package torch
package utils.data.dataset.chunk

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ChunkTensorDataReader,
  ExampleVectorOptional,
  SizeTArrayRef,
  SizeTOptional,
  TensorExampleOptional,
  TensorExampleStack,
  TensorExampleVector,
  ChunkMapTensorDataset as CMTD,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.utils.data.dataset.chunk.ChunkSharedTensorBatchDataset

class ChunkMapTensorDataset(chunkDataset: ChunkSharedTensorBatchDataset)
    extends CMTD(chunkDataset) {

  override def get_batch_example(indices: Long): TensorExampleOptional =
    super.get_batch_example(indices)

  override def size(): SizeTOptional = super.size()

  override def dataset(): pytorch.ChunkSharedTensorBatchDataset = super.dataset()

  override def transform(): TensorExampleStack = super.transform()

  override def get_batch(request: SizeTArrayRef): TensorExampleVector = super.get_batch(request)

  override def get_batch(request: Long*): TensorExampleVector = super.get_batch(request: _*)
}

//  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)
