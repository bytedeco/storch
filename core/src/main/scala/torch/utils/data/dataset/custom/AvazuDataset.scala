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
//class AvazuDataset[Input <: BFloat16 | FloatNN: Default, Target <: BFloat16 | FloatNN: Default]()
//    extends Dataset[Input, Target] {
//  val etledPath = "combined_df.csv"
//
//  var traindf = DataFrame.readCsv(etledPath, 20000000).drop("id") // ,"click")
//  val numCols = traindf.nonnumeric.getColumns
//  var selectDf = traindf.cast(classOf[Float]).numeric
//  var ndArray: NDArray[Float] = selectDf.values[Float]().asInstanceOf[NDArray[Float]]
//  var tensor = Tensor(ndArray)
//  val feature = tensor(::, 0.::(numCols.size)).to(dtype = implicitly[Default[Input]].dtype)
//  val target = tensor(::, numCols.size).to(dtype = implicitly[Default[Target]].dtype)
//
//  override def length: Long = selectDf.length
//
//  override def features: Tensor[Input] = feature
//
//  override def targets: Tensor[Target] = target
//
//  override def getItem(idx: Int): (Tensor[Input], Tensor[Target]) = (feature(idx), target(idx))
//
//  override def iterator: Iterator[(Tensor[Input], Tensor[Target])] = ???
//
//  override def get_batch(request: Long*): ExampleVector = ???
//
//}
