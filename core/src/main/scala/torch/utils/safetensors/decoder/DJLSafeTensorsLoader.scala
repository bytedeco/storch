//package torch.utils.safetensors.decoder
//
//import java.io.{File, FileInputStream, IOException, InputStream}
//import java.nio.ByteBuffer
//import ai.djl.Model
//import ai.djl.ndarray.NDManager
//import ai.djl.ndarray.types.DataType
//import ai.djl.ndarray.types.Shape
//import ai.djl.nn.Parameter
//import torch.utils.safetensors.instance.SafeTensors
//
//class DJLSafeTensorsLoader extends ISafeTensorsLoader[Model] {
//  override def load(tensorsDef: SafeTensors, model: Model): Unit = {
//    val parameterList = model.getBlock.getParameters
//    try {
//      val fis = new FileInputStream(new File(tensorsDef.filename))
//      // val buffer = new Array[Byte](fis.available())
//      // fis.read(buffer)
//      try {
//        // Replace readNBytes with reading into a buffer for compatibility with older Java versions
//        val headerSize = (8 + tensorsDef.sizeOfHeader).toInt
//        val headerBuffer = new Array[Byte](headerSize)
//        fis.read(headerBuffer, 0, headerSize) // skip header
////        fis.readNBytes((8 + tensorsDef.sizeOfHeader).toInt) // skip header
//        val header = tensorsDef.header
////        @Cleanup
//        val nm = NDManager.newBaseManager
//
//        for (element <- header) {
//          val size: Long = element.offsets(1) - element.offsets(0)
//          val sizeInt = size.toInt // Ensure size is an Int for array creation
//          val rawParams = new Array[Byte](sizeInt)
//          fis.read(rawParams, 0, sizeInt)
////          val rawParams = fis.readNBytes(size.toInt)
//          DataType.fromSafetensors(element.dataType.toString)
//          val paramTensor = nm.create(
//            ByteBuffer.wrap(rawParams),
//            new Shape(element.shape*),
//            DataType.fromSafetensors(element.dataType.toString)
//          )
//          val param = Parameter.builder.optArray(paramTensor).build
//          parameterList.add(element.name, param)
//        }
//      } catch { case ex: IOException => throw new RuntimeException(ex.getMessage) }
//      finally if (fis != null) fis.close()
//    }
//  }
//}
//
////import ai.djl.ndarray.NDArray
////import ai.djl.nn.ParameterList
////import lombok.Cleanup
