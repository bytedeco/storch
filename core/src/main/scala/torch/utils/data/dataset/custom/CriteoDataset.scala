//package torch.utils.data.dataset.custom
//
//import org.bytedeco.pytorch.ExampleVector
////import torch.Device.CPU
////import torch.numpy.enums.DType.Float32 as NPFloat32
//import torch.numpy.matrix.NDArray
//import torch.pandas.DataFrame
//import torch.utils.data.Dataset
//import torch.*
//
//class CriteoDataset[Input <: BFloat16 | FloatNN: Default, Target <: BFloat16 | FloatNN: Default]
//    extends Dataset[Input, Target] {
//
//  val path = "criteo_small\\train.txt"
//  val header = Some((0 to 39).map(_.toString).toSeq)
//  val df = DataFrame.readCsv(
//    file = path,
//    separator = "\\t",
//    naString = "",
//    hasHeader = false,
//    limit = -1,
//    needConvert = false,
//    headers = header
//  )
//  val numCols = df.nonnumeric.getColumns
//  var ndArray: NDArray[Float] = df.values[Float]().asInstanceOf[NDArray[Float]]
//  var tensor = Tensor(ndArray)
//  val feature = tensor(::, 0.::(numCols.size)).to(dtype = implicitly[Default[Input]].dtype)
//  val target = tensor(::, numCols.size).to(dtype = implicitly[Default[Target]].dtype)
//
//  override def length: Long = df.length
//
//  override def getItem(idx: Int): (Tensor[Input], Tensor[Target]) = (feature(idx), target(idx))
//
//  override def features: Tensor[Input] = feature
//
//  override def targets: Tensor[Target] = target
//
//  override def iterator: Iterator[(Tensor[Input], Tensor[Target])] = ???
//
//  override def get_batch(request: Long*): ExampleVector = ???
//}
