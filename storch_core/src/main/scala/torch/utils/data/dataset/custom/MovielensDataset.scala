//package torch.utils.data.dataset.custom
//
//import org.bytedeco.pytorch.ExampleVector
//import torch.Device.CPU
//import torch.csv.CSVFormat
////import torch.numpy.enums.DType.Float32 as NPFloat32
////import torch.numpy.matrix.NDArray
//import torch.pandas.DataFrame
//import torch.pandas.component.RatingCSVCompat
//import torch.utils.data.Dataset
//import torch.*
//
//import java.nio.file.Paths
//
//class MovielensDataset[Input <: BFloat16 | FloatNN: Default, Target <: BFloat16 | FloatNN: Default]
//    extends Dataset[Input, Target] {
//
//  val DATA_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
//  val DATA_DIR = "ml-1m"
//  val DATA_PATH = Paths.get(DATA_DIR, "ratings.dat").toString
//  val ZIP_PATH = "ml-1m.zip"
//  given customCSVFormat: CSVFormat = RatingCSVCompat.customCSVFormat
//  val header = Seq("userId", "movieId", "rating", "timestamp")
//  val ratingPath = "D:\\data\\git\\testNumpy\\src\\main\\resources\\ml-1m\\ratings.dat"
//  val moviePath = "D:\\data\\git\\testNumpy\\src\\main\\resources\\ml-1m\\movies.dat"
//  val userPath = "D:\\data\\git\\testNumpy\\src\\main\\resources\\ml-1m\\users.dat"
//  val ratingDf = DataFrame.readRatingCSV(
//    ratingPath,
//    1000,
//    Some(Seq("userId", "movieId", "rating", "timestamp"))
//  ) // (using RatingCSVCompat.customCSVFormat)//.drop("id")//,"click")
//  // MovieID::Title::Genres
//  val movieDf = DataFrame.readRatingCSV(moviePath, -1, Some(Seq("movieId", "title", "genres")))
//  // UserID::Gender::Age::Occupation::Zip-code
//  val userDf = DataFrame.readRatingCSV(
//    userPath,
//    -1,
//    Some(Seq("userId", "gender", "age", "occupation", "zip-code"))
//  )
//  val userIdSeq =
//    ratingDf.columnSelect(Seq("userId")).toArray.flatten.distinct // .unique("userId").show()
//  val movieIdSeq = ratingDf.columnSelect(Seq("movieId")).toArray.flatten.distinct
//
//  val userIdMap =
//    userIdSeq.zipWithIndex.map((userId, index) => (index.toFloat, userId.toString.toInt)).toMap
//  val movieIdMap =
//    movieIdSeq.zipWithIndex.map((movieId, index) => (index.toFloat, movieId.toString.toInt)).toMap
//  println(s" user ${userIdSeq.mkString(",")}, movieIdSeq: ${movieIdSeq.length} ${movieIdSeq
//      .mkString(",")} userIdSeq: ${userIdSeq.length}  movie size ${movieIdSeq.length}")
//
//  val castRatingDF = ratingDf.cast(classOf[Long]).numeric
////  castRatingDF.show()
//  //    System.arraycopy()
//  println(
//    s"castRatingDF ${castRatingDF.getShape}, columns ${castRatingDF.getColumns.mkString(",")}"
//  )
//  val arr = castRatingDF.values[Long](isLazy = true)
//  val userIdNDArray = castRatingDF.columnSelect(Seq("userId")).values[Long](isLazy = false)
//  val movieIdNDArray = castRatingDF.columnSelect(Seq("movieId")).values[Long](isLazy = false)
//  val ratingNDArraySeq = castRatingDF
//    .columnSelect(Seq("rating"))
//    .values[Long](isLazy = false)
//    .trainTestDatasetSplit(0.2, -1, false)
//  val userIdTensor = Tensor[Long](userIdNDArray, false, CPU) // Tensor.fromNDArray(userIdNDArray)
//  val movieIdTensor = Tensor[Long](movieIdNDArray, false, CPU) // Tensor.fromNDArray(movieIdNDArray)
//  val ratingTensor = Tensor[Long](ratingNDArraySeq._1, false, CPU)
//
//  override def length: Long = castRatingDF.length
//
//  override def getItem(idx: Int): (Tensor[Input], Tensor[Target]) = ???
//
//  override def features: Tensor[Input] = ???
//
//  override def targets: Tensor[Target] = ???
//
//  override def iterator: Iterator[(Tensor[Input], Tensor[Target])] = ???
//
//  override def get_batch(request: Long*): ExampleVector = ???
//
//}
