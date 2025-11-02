package torch.utils.data.dataset

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ChunkDatasetOptions,
  InputArchive,
  OutputArchive,
  SizeTOptional,
  TensorExampleVectorOptional,
  ChunkTensorDataset as CTD,
  RandomSampler as RS
}
import torch.utils.data.datareader
import torch.utils.data.datareader.ChunkTensorDataReader

class ChunkTensorDataset(
    chunkReader: datareader.ChunkTensorDataReader,
    chunkSampler: RS,
    batchSampler: RS,
    datasetOptions: ChunkDatasetOptions
) extends CTD(chunkReader, chunkSampler, batchSampler, datasetOptions) {

  override def get_batch(batch_size: Long): TensorExampleVectorOptional =
    super.get_batch(batch_size)

  override def get_batch(): TensorExampleVectorOptional = super.get_batch()

  override def reset(): Unit = super.reset()

  override def size(): SizeTOptional = super.size()

  override def chunk_sampler(): RS = super.chunk_sampler()

  override def save(archive: OutputArchive): Unit = super.save(archive)

  override def load(archive: InputArchive): Unit = super.load(archive)
}

//  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)
