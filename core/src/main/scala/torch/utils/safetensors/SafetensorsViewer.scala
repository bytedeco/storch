package torch.utils.safetensors

import java.io.{
  BufferedInputStream,
  DataInputStream,
  File,
//  FileNotFoundException,
  FileInputStream,
  IOException
}
import java.nio.{ByteBuffer, ByteOrder, FloatBuffer, LongBuffer}
import java.nio.charset.StandardCharsets
import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.util.control.Breaks.break
import scala.util.control.Breaks.breakable
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper

object SafetensorsViewer {

  object HeaderValue {

    def load(jsonNode: JsonNode) = {
      assert(jsonNode.has("dtype"))
      val dtype = jsonNode.get("dtype").asText
      assert(jsonNode.has("shape"))
      val shape = new ListBuffer[Int]
      val jsonNodeShape = jsonNode.get("shape")
      assert(jsonNodeShape.isArray)
      for (i <- 0 until jsonNodeShape.size)
        shape
          .append(jsonNodeShape.get(i).asInt)
      assert(jsonNode.has("data_offsets"))
      val jsonNodeDataOffsets = jsonNode.get("data_offsets")
      assert(jsonNodeDataOffsets.isArray)
      assert(2 == jsonNodeDataOffsets.size)
      val begin = jsonNodeDataOffsets.get(0).asInt
      val end = jsonNodeDataOffsets.get(1).asInt
      val dataOffsets = (begin, end)
      assert(3 == jsonNode.size)
      new SafetensorsViewer.HeaderValue(dtype, shape.toSeq, dataOffsets)
    }
  }

  class HeaderValue(
      val dtype: String,
      val shape: Seq[Int],
      val dataOffsets: (Int, Int)
  ) {
    def getDtype: String = dtype

    def getShape: Seq[Int] = shape

    def getDataOffsets: (Int, Int) = dataOffsets
  }

  @throws[IOException]
  def load(file: File): SafetensorsViewer = {
    var dataInputStream: DataInputStream = null
    val fileInputStream = new FileInputStream(file)
    val bufferedInputStream = new BufferedInputStream(fileInputStream)
    dataInputStream = new DataInputStream(bufferedInputStream)
    val safetensorsViewer = load(dataInputStream)
    dataInputStream.close()
    safetensorsViewer
  }

  @throws[IOException]
  def load(dataInputStream: DataInputStream): SafetensorsViewer = {
    var headerSize = 0L
    val littleEndianBytesHeaderSize = new Array[Byte](8)
    var read = dataInputStream.read(littleEndianBytesHeaderSize)
    assert(8 == read)
    headerSize = ByteBuffer
      .wrap(littleEndianBytesHeaderSize)
      .order(ByteOrder.LITTLE_ENDIAN)
      .getLong

    var stringHeader: String = null
    assert(headerSize <= Integer.MAX_VALUE)
    val bytesHeader = new Array[Byte](headerSize.toInt)
    read = dataInputStream.read(bytesHeader)
    assert(headerSize == read)
    stringHeader = new String(bytesHeader, StandardCharsets.UTF_8)

    val jsonNodeHeader = new ObjectMapper().readTree(stringHeader)
    val header = new mutable.HashMap[String, SafetensorsViewer.HeaderValue]
    val iterator = jsonNodeHeader.fields
    while (iterator.hasNext) {
      val entry = iterator.next
      val tensorName = entry.getKey
      breakable(
        if tensorName == "__metadata__" then break()
      )
      val jsonNode = entry.getValue
      header.put(tensorName, HeaderValue.load(jsonNode))
    }
    var byteBufferSize = 0

    for (headerValue <- header.values)
      byteBufferSize = Math.max(byteBufferSize, headerValue.getDataOffsets._2)
    var byteBuffer: ByteBuffer = null
    val bytes = new Array[Byte](byteBufferSize)
    read = 0
    var total = 0
    breakable(while (0 <= read) {
      read = dataInputStream.read(bytes, total, bytes.length - total)
      total += read
      if (bytes.length == total) break
    })

    assert(byteBufferSize == total)
    byteBuffer = ByteBuffer.wrap(bytes)
    new SafetensorsViewer(header, byteBuffer)
  }
}

class SafetensorsViewer(
    val header: mutable.HashMap[String, SafetensorsViewer.HeaderValue],
    val byteBuffer: ByteBuffer
) {
  private def checkContains(tensorName: String): Unit = {
    if (header.contains(tensorName)) return
    throw new IllegalArgumentException("Tensor not found: " + tensorName)
  }

  def getHeader: mutable.Map[String, SafetensorsViewer.HeaderValue] = header

  def getByteBuffer(tensorName: String): ByteBuffer = {
    checkContains(tensorName)
    val headerValue = header.get(tensorName).get
    val begin = headerValue.dataOffsets._1
    val end = headerValue.getDataOffsets._2
    ByteBuffer
      .wrap(byteBuffer.array, begin, end - begin)
      .order(ByteOrder.LITTLE_ENDIAN)
  }

  def getLongBuffer(tensorName: String): LongBuffer = {
    checkContains(tensorName)
    if (header.get(tensorName).get.getDtype == "I64")
      return getByteBuffer(tensorName).asLongBuffer
    throw new IllegalArgumentException(
      "Unsupported dtype: " + header.get(tensorName).get.getDtype
    )
  }

  def getFloatBuffer(tensorName: String): FloatBuffer = {
    checkContains(tensorName)
    if (header.get(tensorName).get.getDtype == "F32")
      return getByteBuffer(tensorName).asFloatBuffer
    throw new IllegalArgumentException(
      "Unsupported dtype: " + header.get(tensorName).get.getDtype
    )
  }
}
