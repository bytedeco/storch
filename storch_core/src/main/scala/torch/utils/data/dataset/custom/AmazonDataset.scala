//package torch.utils.data.dataset.custom
//
//import org.bytedeco.pytorch.ExampleVector
//import torch.utils.data.Dataset
//import torch.*
//
//import java.nio.file.{Files, Paths}
//
//class AmazonDataset[Input <: BFloat16 | FloatNN: Default, Target <: BFloat16 | FloatNN: Default]
//    extends Dataset[Input, Target] {
//
//  val DATA_URL =
//    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz"
//  val DATA_PATH = "reviews_Electronics_5.json.gz"
//
//  val sc = "reviews_Electronics_5.json.gz"
//
//  val jsonPath = "reviews_Electronics_5.json"
//  val jsonBytes = Files.readAllBytes(Paths.get(jsonPath))
//  val jsonStr = new String(jsonBytes)
//  val data = ujson.read(jsonStr, false).arr
//  data.foreach(println(_))
//
//  override def length: Long = ???
//
//  override def getItem(idx: Int): (Tensor[Input], Tensor[Target]) = ???
//
//  override def features: Tensor[Input] = ???
//
//  override def targets: Tensor[Target] = ???
//
//  override def iterator: Iterator[(Tensor[Input], Tensor[Target])] = ???
//
//  override def get_batch(request: Long*): ExampleVector = ??? // super.get_batch(request)
//}
