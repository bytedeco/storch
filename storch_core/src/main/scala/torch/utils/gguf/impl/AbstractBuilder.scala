package torch.utils.gguf.impl

import torch.utils.gguf.MetadataValueType.*
import torch.utils.gguf.{Builder, MetadataValueType}

abstract class AbstractBuilder extends Builder {
  protected def putValue(key: String, valueType: MetadataValueType, value: Any): BuilderImpl

  protected def putArray(key: String, componentType: MetadataValueType, array: AnyRef): BuilderImpl

//  override def clone(): AnyRef
//  override def clone: AbstractBuilder

  override def containsKey(key: String): Boolean = getValue(classOf[AnyRef], key) != null

  override def containsTensor(tensorName: String): Boolean = getTensor(tensorName) != null

  override def putString(key: String, value: String): Builder =
    putValue(key, MetadataValueType.STRING, value)

  override def putBoolean(key: String, value: Boolean): Builder =
    putValue(key, MetadataValueType.BOOL, value)

  override def putByte(key: String, value: Byte): Builder =
    putValue(key, MetadataValueType.INT8, value)

  override def putUnsignedByte(key: String, value: Byte): Builder =
    putValue(key, MetadataValueType.UINT8, value)

  override def putShort(key: String, value: Short): Builder =
    putValue(key, MetadataValueType.INT16, value)

  override def putUnsignedShort(key: String, value: Short): Builder =
    putValue(key, MetadataValueType.UINT16, value)

  override def putInteger(key: String, value: Int): Builder =
    putValue(key, MetadataValueType.INT32, value)

  override def putUnsignedInteger(key: String, value: Int): Builder =
    putValue(key, MetadataValueType.UINT32, value)

  override def putLong(key: String, value: Long): Builder =
    putValue(key, MetadataValueType.INT64, value)

  override def putUnsignedLong(key: String, value: Long): Builder =
    putValue(key, MetadataValueType.UINT64, value)

  override def putFloat(key: String, value: Float): Builder =
    putValue(key, MetadataValueType.FLOAT32, value)

  override def putDouble(key: String, value: Double): Builder =
    putValue(key, MetadataValueType.FLOAT64, value)

  override def putArrayOfBoolean(key: String, value: Array[Boolean]): Builder =
    putArray(key, MetadataValueType.BOOL, value)

  override def putArrayOfString(key: String, value: Array[String]): Builder =
    putArray(key, MetadataValueType.STRING, value)

  override def putArrayOfByte(key: String, value: Array[Byte]): Builder =
    putArray(key, MetadataValueType.INT8, value)

  override def putArrayOfUnsignedByte(key: String, value: Array[Byte]): Builder =
    putArray(key, MetadataValueType.UINT8, value)

  override def putArrayOfShort(key: String, value: Array[Short]): Builder =
    putArray(key, MetadataValueType.INT16, value)

  override def putArrayOfUnsignedShort(key: String, value: Array[Short]): Builder =
    putArray(key, MetadataValueType.UINT16, value)

  override def putArrayOfInteger(key: String, value: Array[Int]): Builder =
    putArray(key, MetadataValueType.INT32, value)

  override def putArrayOfUnsignedInteger(key: String, value: Array[Int]): Builder =
    putArray(key, MetadataValueType.UINT32, value)

  override def putArrayOfLong(key: String, value: Array[Long]): Builder =
    putArray(key, MetadataValueType.INT64, value)

  override def putArrayOfUnsignedLong(key: String, value: Array[Long]): Builder =
    putArray(key, MetadataValueType.UINT64, value)

  override def putArrayOfFloat(key: String, value: Array[Float]): Builder =
    putArray(key, MetadataValueType.FLOAT32, value)

  override def putArrayOfDouble(key: String, value: Array[Double]): Builder =
    putArray(key, MetadataValueType.FLOAT64, value)

}
