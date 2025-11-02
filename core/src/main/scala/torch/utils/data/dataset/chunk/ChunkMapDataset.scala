package torch
package utils.data.dataset.chunk

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ChunkDataReader,
  ExampleOptional,
  ExampleStack,
  ExampleVector,
  ExampleVectorOptional,
  SizeTArrayRef,
  SizeTOptional,
  ChunkMapDataset as CMD,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.utils.data.dataset.chunk.ChunkSharedBatchDataset

class ChunkMapDataset(chunkDataset: ChunkSharedBatchDataset) extends CMD(chunkDataset) {

  override def get_batch_example(indices: Long): ExampleOptional = super.get_batch_example(indices)

  override def size(): SizeTOptional = super.size()

  override def dataset(): pytorch.ChunkSharedBatchDataset = super.dataset()

  override def transform(): ExampleStack = super.transform()

  override def get_batch(request: SizeTArrayRef): ExampleVector = super.get_batch(request)

  override def get_batch(request: Long*): ExampleVector = super.get_batch(request: _*)
}

//  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)

//  override def get_batch(request: SizeTArrayRef): ExampleVector = super.get_batch(request)
