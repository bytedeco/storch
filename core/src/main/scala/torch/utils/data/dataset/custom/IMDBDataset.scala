package torch.utils.data.dataset.custom

import org.bytedeco.pytorch.ExampleVector
import torch.utils.data.Dataset
import torch.*

import java.nio.file.Paths

class IMDBDataset[Input <: BFloat16 | FloatNN: Default, Target <: BFloat16 | FloatNN: Default]
    extends Dataset[Input, Target] {

  val DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
  val DATA_ROOT = "./data"
  val tarGzPath = Paths.get(DATA_ROOT, "aclImdb_v1.tar.gz")
  val dataDir = Paths.get(DATA_ROOT, "aclImdb")

  override def length: Long = ???

  override def getItem(idx: Int): (Tensor[Input], Tensor[Target]) = ???

  override def features: Tensor[Input] = ???

  override def targets: Tensor[Target] = ???

  override def iterator: Iterator[(Tensor[Input], Tensor[Target])] = ???

  override def get_batch(request: Long*): ExampleVector = ???

}
