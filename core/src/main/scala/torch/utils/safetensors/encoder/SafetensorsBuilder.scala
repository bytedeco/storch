package torch.utils.safetensors.encoder

import java.io.*
import java.nio.charset.StandardCharsets
import java.nio.{ByteBuffer, ByteOrder, FloatBuffer, LongBuffer}
import scala.collection.mutable

object SafetensorsBuilder {
  class HeaderValue(
      val dtype: String,
      val shape: Seq[Int],
      val dataOffsets: (Int, Int)
  ) {
    def serialize = {
      val shapeBuilder = new StringBuilder
      shapeBuilder.append('[')
      if (!shape.isEmpty) {
        for (i <- shape) {
          shapeBuilder.append(i)
          shapeBuilder.append(',')
        }
        shapeBuilder.deleteCharAt(shapeBuilder.length - 1)
      }
      shapeBuilder.append(']')
      val result = "{'dtype':'%s','shape':%s,'data_offsets':[%d,%d]}"
        .replaceAll("'", "\"")
      String.format(result, dtype, shapeBuilder, dataOffsets._1, dataOffsets._2)
    }
  }

  def checkLength(shape: Seq[Int], length: Int): Unit = {
    var expect = 1

    for (i <- shape) expect *= i
    if (expect == length) return
    throw new IllegalArgumentException(
      "Shape does not match length: " + shape + "," + length
    )
  }
}

class SafetensorsBuilder {
  private final val header =
    new mutable.HashMap[String, SafetensorsBuilder.HeaderValue]
  private final val bodies = new mutable.HashMap[String, AnyRef]
  private var byteSize = 0

  def add(tensorName: String, shape: Seq[Int], longs: Array[Long]): Unit = {
    SafetensorsBuilder.checkLength(shape, longs.length)

    val begin = byteSize
    byteSize += java.lang.Long.BYTES * longs.length
    val end = byteSize
    val dataOffsets = (begin, end)
    val headerValue =
      new SafetensorsBuilder.HeaderValue("I64", shape, dataOffsets)
    header.put(tensorName, headerValue)
    bodies.put(tensorName, longs)
  }

  def add(tensorName: String, shape: Seq[Int], longBuffer: LongBuffer): Unit = {
    SafetensorsBuilder.checkLength(shape, longBuffer.limit)

    val begin = byteSize
    byteSize += java.lang.Long.BYTES * (longBuffer.limit - longBuffer.position)
    val end = byteSize
    val dataOffsets = (begin, end)
    val headerValue =
      new SafetensorsBuilder.HeaderValue("I64", shape, dataOffsets)
    header.put(tensorName, headerValue)
    bodies.put(tensorName, longBuffer)
  }

  def add(tensorName: String, shape: Seq[Int], floats: Array[Float]): Unit = {
    SafetensorsBuilder.checkLength(shape, floats.length)

    val begin = byteSize
    byteSize += java.lang.Float.BYTES * floats.length
    val end = byteSize
    val dataOffsets = (begin, end)
    val headerValue =
      new SafetensorsBuilder.HeaderValue("F32", shape, dataOffsets)
    header.put(tensorName, headerValue)
    bodies.put(tensorName, floats)
  }

  def add(tensorName: String, shape: Seq[Int], floatBuffer: FloatBuffer): Unit = {
    SafetensorsBuilder.checkLength(shape, floatBuffer.limit)

    val begin = byteSize
    byteSize +=
      java.lang.Float.BYTES * (floatBuffer.limit - floatBuffer.position)
    val end = byteSize
    val dataOffsets = (begin, end)
    val headerValue =
      new SafetensorsBuilder.HeaderValue("F32", shape, dataOffsets)
    header.put(tensorName, headerValue)
    bodies.put(tensorName, floatBuffer)
  }

  def contentLength: Int = java.lang.Long.BYTES +
    serializeHeader.getBytes(StandardCharsets.UTF_8).length + byteSize

  private def serializeHeader = {
    val headerBuilder = new StringBuilder
    headerBuilder.append('{')
    if (!header.isEmpty) {
      for (entry <- header) {
        headerBuilder.append('"')
        headerBuilder.append(entry._1)
        headerBuilder.append('"')
        headerBuilder.append(':')
        headerBuilder.append(entry._2.serialize)
        headerBuilder.append(',')
      }
      headerBuilder.deleteCharAt(headerBuilder.length - 1)
    }
    headerBuilder.append('}')
    val stringHeader = headerBuilder.toString
    val padding = 8 - stringHeader.getBytes(StandardCharsets.UTF_8).length % 8

    headerBuilder.append(" ".*(padding))
    headerBuilder.toString
  }

  private def serializeByteBuffer = {
    val byteBuffer = ByteBuffer.wrap(new Array[Byte](byteSize))
    for (entry <- header) {
      var bb: ByteBuffer = null
      val dataOffsets = entry._2.dataOffsets
      val begin = dataOffsets._1
      val end = dataOffsets._2
      bb = ByteBuffer
        .wrap(byteBuffer.array, begin, end - begin)
        .order(ByteOrder.LITTLE_ENDIAN)
      val instance = bodies.getOrElse(entry._1, "") // .asInstanceOf[Array[Long]]
      println(instance.getClass.getTypeName)
      instance match {
        case arr: Array[Long] =>
          println("instance is match Array[Long]")
          bb.asLongBuffer.put(arr)
        case buf: LongBuffer =>
          println("instance is match LongBuffer")
          bb.asLongBuffer.put(buf)
        case arr: Array[Float] =>
          println("instance is match Array[Float]")
          bb.asFloatBuffer.put(arr)
        case buf: FloatBuffer =>
          println("instance is match FloatBuffer")
          bb.asFloatBuffer.put(buf)
        case _ =>
          throw new IllegalArgumentException(
            "Unsupported type: " + instance.getClass.getTypeName
          )
      }
    }
    byteBuffer
  }

  @throws[IOException]
  def save(file: File): Unit = {
    var dataOutputStream: DataOutputStream = null
    var fileOutputStream: FileOutputStream = null
    try fileOutputStream = new FileOutputStream(file)
    catch { case e: FileNotFoundException => throw new IOException(e) }
    val bufferedOutputStream = new BufferedOutputStream(fileOutputStream)
    dataOutputStream = new DataOutputStream(bufferedOutputStream)
    save(dataOutputStream)
    dataOutputStream.close()
  }

  @throws[IOException]
  def save(dataOutputStream: DataOutputStream): Unit = {
    val stringHeader = serializeHeader
    val littleEndianBytesHeaderSize = new Array[Byte](java.lang.Long.BYTES)
    val headerSize = stringHeader.getBytes(StandardCharsets.UTF_8).length
    ByteBuffer
      .wrap(littleEndianBytesHeaderSize)
      .order(ByteOrder.LITTLE_ENDIAN)
      .asLongBuffer
      .put(headerSize)
    dataOutputStream.write(littleEndianBytesHeaderSize)
    dataOutputStream.writeBytes(stringHeader)
    dataOutputStream.write(serializeByteBuffer.array)
  }
}
