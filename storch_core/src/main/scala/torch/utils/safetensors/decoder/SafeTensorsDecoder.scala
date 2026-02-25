package torch.utils.safetensors.decoder

import java.io.{File, FileInputStream, IOException}
import java.nio.file.Paths
import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.jdk.CollectionConverters.*
import com.alibaba.fastjson2.JSON
import torch.utils.safetensors.instance.SafeTensors
import scala.util.control.Breaks.{break, breakable}

/** SafeTensors format parser
  */

//@RequiredArgsConstructor
class SafeTensorsDecoder(filename: String) {

  private def getSizeOfHeader(sizeOfHeader: Array[Byte]) = {
    var size = 0
    for (i <- 7 to 0 by -1) {
      size <<= 8
      size = size | sizeOfHeader(i) & 0xff
    }
    size
  }

  def decode: SafeTensors =
    try {
      val fis: FileInputStream = new FileInputStream(new File(filename))
      try {
//        val headerSize: Long = getSizeOfHeader(fis.readNBytes(8))
//        val headerJsonBytes = fis.readNBytes(headerSize.toInt)
        // Replace readNBytes with reading into a buffer for compatibility with older Java versions
        val sizeOfHeaderBytes = new Array[Byte](8)
        fis.read(sizeOfHeaderBytes, 0, 8)
        val headerSize: Long = getSizeOfHeader(sizeOfHeaderBytes)

        // Replace readNBytes with reading into a buffer for compatibility with older Java versions
        val headerJsonBytes = new Array[Byte](headerSize.toInt)
        fis.read(headerJsonBytes, 0, headerSize.toInt)
        val headerJson = JSON.parseObject(headerJsonBytes)
        val safeTensors = new SafeTensors
        safeTensors.filename = Paths.get(filename).toFile.getAbsolutePath
        safeTensors.sizeOfHeader = headerSize
        val elements = new ListBuffer[SafeTensors.HeaderElement]()
        for (name <- headerJson.asScala.keySet) {
          breakable {
            if ("__metadata__" == name) {
              val metadata = headerJson.getJSONObject("__metadata__").asScala
//                .toJavaObject(classOf[mutable.Map[_, _]])
              safeTensors.metadata = metadata.map((k, v) => (k, v.toString))
//              continue // todo: continue is not supported
              break
            }
          }

          val header = headerJson.getJSONObject(name)
          val dataType = header.getString("dtype")
          val shape = header.getList("shape", classOf[Long]).asScala.toSeq
          val offsets = header.getList("data_offsets", classOf[Long]).asScala.toSeq
          val element = new SafeTensors.HeaderElement(
            name = name,
            dataType = SafeTensors.DataType.valueOf(dataType),
            shape = shape,
            offsets = offsets
          )
          elements.append(element)
        }
        safeTensors.header = elements
        safeTensors
      } catch { case ex: IOException => throw new RuntimeException(ex.getMessage) }
      finally if (fis != null) fis.close()
    }
}

//import com.alibaba.fastjson2.JSONObject
//import lombok.NonNull
//import lombok.RequiredArgsConstructor
