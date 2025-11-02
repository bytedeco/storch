package torch.utils.data.dataset

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ChunkDataReader,
  ChunkDatasetOptions,
  ExampleVectorOptional,
  InputArchive,
  OutputArchive,
  Sampler,
  SizeTOptional,
  ChunkDataset as CDS,
  RandomSampler as RS,
  SequentialSampler as SS
}

class ChunkDataset(
    chunkReader: ChunkDataReader,
    chunkSampler: RS,
    batchSampler: RS,
    datasetOptions: ChunkDatasetOptions
) extends CDS(chunkReader, chunkSampler, batchSampler, datasetOptions) {

  override def get_batch(batch_size: Long): ExampleVectorOptional =
    super.get_batch(datasetOptions.batch_size().get())

  override def get_batch(): ExampleVectorOptional = super.get_batch()

  override def reset(): Unit = super.reset()

  override def size(): SizeTOptional = super.size()

  override def chunk_sampler(): RS = super.chunk_sampler()

  override def save(archive: OutputArchive): Unit = super.save(archive)

  override def load(archive: InputArchive): Unit = super.load(archive)
}
