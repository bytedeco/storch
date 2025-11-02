package torch.utils.data.dataset.custom

import org.bytedeco.pytorch.ExampleVector
import torch.utils.data.Dataset
import torch.*

class YelpDataset[Input <: BFloat16 | FloatNN: Default, Target <: BFloat16 | FloatNN: Default]
    extends Dataset[Input, Target] {
  val yelpPhotoPath = "https://business.yelp.com/external-assets/files/Yelp-Photos.zip"
//https://business.yelp.com/data/resources/open-dataset/
  val yelpJsonPath = "https://business.yelp.com/external-assets/files/Yelp-JSON.zip"
  val DATA_URL =
    "https://snap.stanford.edu/data/yelp-dataset/yelp_academic_dataset_review.json.gz"

  override def length: Long = ???

  override def getItem(idx: Int): (Tensor[Input], Tensor[Target]) = ???

  override def features: Tensor[Input] = ???

  override def targets: Tensor[Target] = ???

  override def iterator: Iterator[(Tensor[Input], Tensor[Target])] = ???

  override def get_batch(request: Long*): ExampleVector = ???

}
