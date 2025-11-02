package torch.utils.gguf.impl

import torch.utils.gguf.MetadataValueType.*
import torch.utils.gguf.{GGMLType, GGUF, MetadataValueType, TensorInfo}

import java.io.IOException
import java.lang.reflect.Array as JArray
import java.nio.channels.WritableByteChannel
import java.nio.charset.StandardCharsets
import java.nio.{ByteBuffer, ByteOrder}
import scala.jdk.CollectionConverters.*

object WriterImpl {
  @throws[IOException]
  private[impl] def writeImpl(gguf: GGUF, byteChannel: WritableByteChannel): Unit = {
    val writer = new WriterImpl(gguf)
    writer.writeHeader(byteChannel)
    for (tensorInfo <- gguf.getTensors) {
      writer.writeTensorInfo(byteChannel, tensorInfo)
    }
    // Always align, even if there are no tensors.
    writer.writePaddingForAlignment(byteChannel)
    assert(writer.totalBytesWritten == gguf.getTensorDataOffset)
  }
}

final class WriterImpl(val gguf: GGUF) {
  final private val BB_8: ByteBuffer =
    ByteBuffer.allocate(java.lang.Long.BYTES).order(ByteOrder.nativeOrder)
  private var totalBytesWritten = 0L

  @throws[IOException]
  private def writeFully(byteChannel: WritableByteChannel, byteBuffer: ByteBuffer): Unit = {
    while (byteBuffer.hasRemaining) this.totalBytesWritten += byteChannel.write(byteBuffer)
  }

  @throws[IOException]
  private def writePaddingForAlignment(byteChannel: WritableByteChannel): Unit = {
    val padding = GGUFImpl.padding(this.totalBytesWritten, gguf.getAlignment).toInt
    writeFully(byteChannel, ByteBuffer.allocate(padding))
  }

  @throws[IOException]
  private def writeLong(byteChannel: WritableByteChannel, value: Long): Unit = {
    writeFully(
      byteChannel,
      BB_8.clear.asInstanceOf[ByteBuffer].putLong(value).flip.asInstanceOf[ByteBuffer]
    )
  }

  @throws[IOException]
  private def writeDouble(byteChannel: WritableByteChannel, value: Double): Unit = {
    writeLong(byteChannel, java.lang.Double.doubleToRawLongBits(value))
  }

  @throws[IOException]
  private def writeInt(byteChannel: WritableByteChannel, value: Int): Unit = {
    writeFully(
      byteChannel,
      BB_8.clear.asInstanceOf[ByteBuffer].putInt(value).flip.asInstanceOf[ByteBuffer]
    )
  }

  @throws[IOException]
  private def writeFloat(byteChannel: WritableByteChannel, value: Float): Unit = {
    writeInt(byteChannel, java.lang.Float.floatToRawIntBits(value))
  }

  @throws[IOException]
  private def writeByte(byteChannel: WritableByteChannel, value: Byte): Unit = {
    writeFully(
      byteChannel,
      BB_8.clear.asInstanceOf[ByteBuffer].put(value).flip.asInstanceOf[ByteBuffer]
    )
  }

  @throws[IOException]
  private def writeBoolean(byteChannel: WritableByteChannel, value: Boolean): Unit = {
    writeByte(
      byteChannel,
      if (value) 1.toByte
      else 0.toByte
    )
  }

  @throws[IOException]
  private def writeShort(byteChannel: WritableByteChannel, value: Short): Unit = {
    writeFully(
      byteChannel,
      BB_8.clear.asInstanceOf[ByteBuffer].putShort(value).flip.asInstanceOf[ByteBuffer]
    )
  }

  @throws[IOException]
  private def writeBytes(byteChannel: WritableByteChannel, bytes: Array[Byte]): Unit = {
    writeFully(byteChannel, ByteBuffer.wrap(bytes))
  }

  @throws[IOException]
  private def writeString(byteChannel: WritableByteChannel, string: String): Unit = {
    val bytes = string.getBytes(StandardCharsets.UTF_8)
    // A string in GGUF.
    // The length of the string, in bytes.
    writeLong(byteChannel, bytes.length) // uint64_t len;

    // The string as a UTF-8 non-null-terminated string.
    writeBytes(byteChannel, bytes)
  }

  @throws[IOException]
  private def writeTensorInfo(byteChannel: WritableByteChannel, tensorInfo: TensorInfo): Unit = {
    // The name of the tensor. It is a standard GGUF string, with the caveat that
    // it must be at most 64 bytes long.
    val name = tensorInfo.name
    assert(name.length <= 64)
    writeString(byteChannel, name) // gguf_string_t name;
    // The number of shape in the tensor.
    // Currently at most 4, but this may change in the future.
    val n_dimensions = tensorInfo.shape.length
    assert(n_dimensions <= 4)
    writeInt(byteChannel, n_dimensions) // uint32_t n_dimensions;
    // The shape of the tensor.
    val dimensions = tensorInfo.shape // uint64_t shape[n_dimensions];
    for (i <- 0 until n_dimensions) {
      assert(dimensions(i) > 0)
      writeLong(byteChannel, dimensions(i))
    }
    // The type of the tensor.
    val ggmlType = tensorInfo.ggmlType
    writeGGMLType(byteChannel, ggmlType) // ggml_type type;
    // The offset of the tensor's data in this file in bytes.
    // This offset is relative to `tensor_data`, not to the start
    // of the file, to make it easier for writers to write the file.
    // Readers should consider exposing this offset relative to the
    // file to make it easier to read the data.
    // Must be a multiple of `ALIGNMENT`.
    val offset = tensorInfo.offset
    writeLong(byteChannel, offset) // uint64_t offset;
    assert(offset % gguf.getAlignment == 0)
    // return new GGUFTensorInfo(name, dimensions, ggmlType, offset);
  }

  @SuppressWarnings(Array("EnumOrdinal"))
  @throws[IOException]
  private def writeGGMLType(byteChannel: WritableByteChannel, ggmlType: GGMLType): Unit = {
    writeInt(byteChannel, ggmlType.ordinal)
  }

  @throws[IOException]
  private def writeHeader(byteChannel: WritableByteChannel): Unit = {
    // Magic number to announce that this is a GGUF file.
    // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
    // Your executor might do little-endian byte order, so it might be
    // check for 0x46554747 and letting the endianness cancel out.
    // Consider being *very* explicit about the byte order here.
    writeInt(byteChannel, ReaderImpl.GGUF_MAGIC) //    uint32_t magic;

    // The version of the format implemented.
    // Must be `3` for version described in this spec.
    //
    // This version should only be increased for structural changes to the format.
    // Changes that do not affect the structure of the file should instead update the metadata
    // to signify the change.
    writeInt(byteChannel, gguf.getVersion) // uint32_t version;

    // The number of tensors in the file.
    // This is explicit, instead of being included in the metadata, to ensure it is always present
    // for loading the tensors.
    writeLong(byteChannel, gguf.getTensors.size) // uint64_t tensor_count;

    // The number of metadata key-value pairs.
    writeLong(byteChannel, gguf.getMetadataKeys.size) // uint64_t metadata_kv_count;

    // The metadata key-value pairs.
    // gguf_metadata_kv_t metadata_kv[metadata_kv_count];
    for (key <- gguf.getMetadataKeys) {
      // The key of the metadata. It is a standard GGUF string, with the following caveats:
      // - It must be a valid ASCII string.
      // - It must be a hierarchical key, where each segment is `lower_snake_case` and separated by
      // a `.`.
      // - It must be at most 2^16-1/65535 bytes long.
      // Any keys that do not follow these rules are invalid.
      assert(key.length < (1 << 16))
      assert(
        key.codePoints.allMatch((cp: Int) =>
          ('a' <= cp && cp <= 'z') || ('0' <= cp && cp <= '9') || cp == '_' || cp == '.'
        )
      )
      writeString(byteChannel, key)
      val value = gguf.getValue(classOf[AnyRef], key)
      assert(value != null)
      val valueType = gguf.getType(key)
      assert(valueType != null)
      if (valueType eq MetadataValueType.ARRAY) {
        val componentType = gguf.getComponentType(key)
        writeTypedArrayOf(byteChannel, componentType, value)
      } else writeTypedValue(byteChannel, valueType, value)
    }
  }

  @throws[IOException]
  private def writeTypedArrayOf(
      byteChannel: WritableByteChannel,
      componentType: MetadataValueType,
      value: AnyRef
  ): Unit = {
    val arrayLength = JArray.getLength(value)
    writeValueType(byteChannel, MetadataValueType.ARRAY)
    writeValueType(byteChannel, componentType)
    writeLong(byteChannel, arrayLength)
    componentType match {
//      case UINT8 => // fall-through
      case INT8 | UINT8 =>
        writeBytes(byteChannel, value.asInstanceOf[Array[Byte]])
//      case UINT16 => // fall-through
      case INT16 | UINT16 =>
        for (s <- value.asInstanceOf[Array[Short]]) {
          writeShort(byteChannel, s)
        }
//      case UINT32 => // fall-through
      case INT32 | UINT32 =>
        for (i <- value.asInstanceOf[Array[Int]]) {
          writeInt(byteChannel, i)
        }
      case BOOL =>
        for (b <- value.asInstanceOf[Array[Boolean]]) {
          writeBoolean(byteChannel, b)
        }
      case STRING =>
        val stringArray = value.asInstanceOf[Array[String]]
        for (s <- stringArray) {
          writeString(byteChannel, s)
        }
//      case UINT64 => // fall-through
      case INT64 | UINT64 =>
        for (n <- value.asInstanceOf[Array[Long]]) {
          writeLong(byteChannel, n)
        }
      case FLOAT32 =>
        for (f <- value.asInstanceOf[Array[Float]]) {
          writeFloat(byteChannel, f)
        }
      case FLOAT64 =>
        for (d <- value.asInstanceOf[Array[Double]]) {
          writeDouble(byteChannel, d)
        }
      case ARRAY =>
        throw new UnsupportedOperationException("array of arrays")
    }
  }

  @throws[IOException]
  private def writeTypedValue(
      byteChannel: WritableByteChannel,
      valueType: MetadataValueType,
      value: AnyRef
  ): Unit = {
    if (valueType eq MetadataValueType.ARRAY)
      throw new IllegalArgumentException("use writeArrayOf instead")
    writeValueType(byteChannel, valueType)
    valueType match {
//      case UINT8 => // fall-through
      case INT8 | UINT8 =>
        writeByte(byteChannel, value.asInstanceOf[Byte])
//      case UINT16 => // fall-through
      case INT16 | UINT16 =>
        writeShort(byteChannel, value.asInstanceOf[Short])
//      case UINT32 => // fall-through
      case INT32 | UINT32 =>
        writeInt(byteChannel, value.asInstanceOf[Int])
      case FLOAT32 =>
        writeFloat(byteChannel, value.asInstanceOf[Float])
      case BOOL =>
        writeBoolean(byteChannel, value.asInstanceOf[Boolean])
      case STRING =>
        writeString(byteChannel, value.asInstanceOf[String])
      case ARRAY =>
        throw new IllegalArgumentException("use writeArrayOf instead")
//      case UINT64 => // fall-through
      case INT64 | UINT64 =>
        writeLong(byteChannel, value.asInstanceOf[Long])
      case FLOAT64 =>
        writeDouble(byteChannel, value.asInstanceOf[Double])
    }
  }

  @SuppressWarnings(Array("EnumOrdinal"))
  @throws[IOException]
  private def writeValueType(
      byteChannel: WritableByteChannel,
      valueType: MetadataValueType
  ): Unit = {
    writeInt(byteChannel, valueType.ordinal)
  }
}
