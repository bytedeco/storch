package torch.utils.gguf

import torch.utils.gguf.impl.ImplAccessor

//import java.util

/** Builder interface for the GGUF format.
  *
  * @see
  *   <a href="https://github.com/ggerganov/ggml/blob/master/docs/gguf.md">GGUF format
  *   specification</a>
  */
object Builder {

  /** Creates a new {@link Builder} from an existing GGUF instance.
    */
  def newBuilder(gguf: GGUF): Builder = ImplAccessor.newBuilder(gguf)

  /** Creates a new empty {@link Builder} .
    */
  def newBuilder: Builder = ImplAccessor.newBuilder
}

trait Builder {

  /** Builds a GGUF instance with automatic tensor offset computation.
    */
  def build: GGUF = build(true)

  /** Builds a GGUF instance.
    *
    * @param recomputeTensorOffsets
    *   if true, tensor offsets will be automatically re-computed, packed in the same order and
    *   respecting the alignment
    */
  def build(recomputeTensorOffsets: Boolean): GGUF

  /** Creates and returns a copy of this object.
    */
//  def clone: Builder

  /** Sets the GGUF format version.
    */
  def setVersion(newVersion: Int): Builder

  /** Gets the GGUF format version.
    */
  def getVersion: Int

  /** Sets the alignment value for tensor data.
    *
    * @throws IllegalArgumentException
    *   if alignment is not a power of 2
    */
  def setAlignment(newAlignment: Int): Builder = {
    if (newAlignment < 0 || Integer.bitCount(newAlignment) != 1)
      throw new IllegalArgumentException("alignment must be a power of 2 but was " + newAlignment)
    putUnsignedInteger(ImplAccessor.alignmentKey, newAlignment)
  }

  /** Gets the current alignment value or the default if not set.
    */
  def getAlignment: Int = {
    if (containsKey(ImplAccessor.alignmentKey)) {
      assert(getType(ImplAccessor.alignmentKey) eq MetadataValueType.UINT32)
      return getValue(classOf[Int], ImplAccessor.alignmentKey)
    }
    ImplAccessor.defaultAlignment
  }

  /** Adds or updates a tensor.
    */
  def putTensor(tensorInfo: TensorInfo): Builder

  /** Removes a tensor by name.
    */
  def removeTensor(tensorName: String): Builder

  /** Checks if a tensor exists by name.
    */
  def containsTensor(tensorName: String): Boolean

  /** Gets tensor information by name.
    */
  def getTensor(tensorName: String): TensorInfo

  /** Checks if a metadata key exists.
    */
  def containsKey(key: String): Boolean

  /** Gets a metadata value associated with the given key, casting it to the specified target class,
    * or null if the key is not found.
    *
    * @see
    *   GGUF#getValue(Class, String)
    */
  def getValue[T](targetClass: Class[T], key: String): T

  /** Gets all metadata keys, order is preserved.
    *
    * @return
    *   the set of metadata keys
    */
  def getMetadataKeys: Set[String]

  /** Gets all tensors, order is preserved.
    *
    * @return
    *   the collection of tensor information
    */
  def getTensors: Seq[TensorInfo]

  /** Gets the component type for the array value associated with the given key.
    *
    * @param key
    *   the key to look up
    * @return
    *   the component type, or null if the key doesn't exist or value is not an array
    */
  def getComponentType(key: String): MetadataValueType

  /** Gets the type of the metadata value associated with the given key.
    *
    * @param key
    *   the key to look up
    * @return
    *   the metadata value type, or null if the key doesn't exist
    */
  def getType(key: String): MetadataValueType

  /** Removes a metadata key.
    *
    * @param key
    *   the key to remove
    * @return
    *   this builder instance
    */
  def removeKey(key: String): Builder

  /** Sets a String for the given metadata key. Value type: {@link MetadataValueType# STRING}
    *
    * @param key
    *   the key to associate the value with
    * @param value
    *   the string value to set
    * @return
    *   this builder instance
    */
  def putString(key: String, value: String): Builder

  /** Sets a boolean for the given metadata key. Value type: {@link MetadataValueType# BOOL}
    *
    * @param key
    *   the key to associate the value with
    * @param value
    *   the boolean value to set
    * @return
    *   this builder instance
    */
  def putBoolean(key: String, value: Boolean): Builder

  /** Sets a byte for the given metadata key. Value type: {@link MetadataValueType# INT8}
    *
    * @param key
    *   the key to associate the value with
    * @param value
    *   the byte value to set
    * @return
    *   this builder instance
    */
  def putByte(key: String, value: Byte): Builder

  /** Sets an unsigned byte for the given metadata key. Value type: {@link MetadataValueType# UINT8}
    *
    * @param key
    *   the key to associate the value with
    * @param value
    *   the unsigned byte value to set
    * @return
    *   this builder instance
    */
  def putUnsignedByte(key: String, value: Byte): Builder

  /** Sets a short for the given metadata key. Value type: {@link MetadataValueType# INT16}
    *
    * @param key
    *   the key to associate the value with
    * @param value
    *   the short value to set
    * @return
    *   this builder instance
    */
  def putShort(key: String, value: Short): Builder

  /** Sets an unsigned short for the given metadata key. Value type: {@link MetadataValueType#
    * UINT16}
    *
    * @param key
    *   the key to associate the value with
    * @param value
    *   the unsigned short value to set
    * @return
    *   this builder instance
    */
  def putUnsignedShort(key: String, value: Short): Builder

  /** Sets an integer for the given metadata key. Value type: {@link MetadataValueType# INT32}
    *
    * @param key
    *   the key to associate the value with
    * @param value
    *   the integer value to set
    * @return
    *   this builder instance
    */
  def putInteger(key: String, value: Int): Builder

  /** Sets an unsigned integer for the given metadata key. Value type: {@link MetadataValueType#
    * UINT32}
    *
    * @param key
    *   the key to associate the value with
    * @param value
    *   the unsigned integer value to set
    * @return
    *   this builder instance
    */
  def putUnsignedInteger(key: String, value: Int): Builder

  /** Sets a long for the given metadata key. Value type: {@link MetadataValueType# INT64}
    *
    * @param key
    *   the key to associate the value with
    * @param value
    *   the long value to set
    * @return
    *   this builder instance
    */
  def putLong(key: String, value: Long): Builder

  /** Sets an unsigned long for the given metadata key. Value type: {@link MetadataValueType#
    * UINT64}
    *
    * @param key
    *   the key to associate the value with
    * @param value
    *   the unsigned long value to set
    * @return
    *   this builder instance
    */
  def putUnsignedLong(key: String, value: Long): Builder

  /** Sets a float for the given metadata key. Value type: {@link MetadataValueType# FLOAT32}
    *
    * @param key
    *   the key to associate the value with
    * @param value
    *   the float value to set
    * @return
    *   this builder instance
    */
  def putFloat(key: String, value: Float): Builder

  /** Sets a double for the given metadata key. Value type: {@link MetadataValueType# FLOAT64}
    *
    * @param key
    *   the key to associate the value with
    * @param value
    *   the double value to set
    * @return
    *   this builder instance
    */
  def putDouble(key: String, value: Double): Builder

  /** Sets a boolean array for the given metadata key. Value type: {@link MetadataValueType# ARRAY}
    * Component type: {@link MetadataValueType# BOOL}
    *
    * @param key
    *   the key to associate the array with
    * @param value
    *   the boolean array to set
    * @return
    *   this builder instance
    */
  def putArrayOfBoolean(key: String, value: Array[Boolean]): Builder

  /** Sets a String array for the given metadata key. Value type: {@link MetadataValueType# ARRAY}
    * Component type: {@link MetadataValueType# STRING}
    *
    * @param key
    *   the key to associate the array with
    * @param value
    *   the string array to set
    * @return
    *   this builder instance
    */
  def putArrayOfString(key: String, value: Array[String]): Builder

  /** Sets a byte array for the given metadata key. Value type: {@link MetadataValueType# ARRAY}
    * Component type: {@link MetadataValueType# INT8}
    *
    * @param key
    *   the key to associate the array with
    * @param value
    *   the byte array to set
    * @return
    *   this builder instance
    */
  def putArrayOfByte(key: String, value: Array[Byte]): Builder

  /** Sets an unsigned byte array for the given metadata key. Value type: {@link MetadataValueType#
    * ARRAY} Component type: {@link MetadataValueType# UINT8}
    *
    * @param key
    *   the key to associate the array with
    * @param value
    *   the unsigned byte array to set
    * @return
    *   this builder instance
    */
  def putArrayOfUnsignedByte(key: String, value: Array[Byte]): Builder

  /** Sets a short array for the given metadata key. Value type: {@link MetadataValueType# ARRAY}
    * Component type: {@link MetadataValueType# INT16}
    *
    * @param key
    *   the key to associate the array with
    * @param value
    *   the short array to set
    * @return
    *   this builder instance
    */
  def putArrayOfShort(key: String, value: Array[Short]): Builder

  /** Sets an unsigned short array for the given metadata key. Value type: {@link MetadataValueType#
    * ARRAY} Component type: {@link MetadataValueType# UINT16}
    *
    * @param key
    *   the key to associate the array with
    * @param value
    *   the unsigned short array to set
    * @return
    *   this builder instance
    */
  def putArrayOfUnsignedShort(key: String, value: Array[Short]): Builder

  /** Sets an integer array for the given metadata key. Value type: {@link MetadataValueType# ARRAY}
    * Component type: {@link MetadataValueType# INT32}
    *
    * @param key
    *   the key to associate the array with
    * @param value
    *   the integer array to set
    * @return
    *   this builder instance
    */
  def putArrayOfInteger(key: String, value: Array[Int]): Builder

  /** Sets an unsigned integer array for the given metadata key. Value type: {@link
    * MetadataValueType# ARRAY} Component type: {@link MetadataValueType# UINT32}
    *
    * @param key
    *   the key to associate the array with
    * @param value
    *   the unsigned integer array to set
    * @return
    *   this builder instance
    */
  def putArrayOfUnsignedInteger(key: String, value: Array[Int]): Builder

  /** Sets a long array for the given metadata key. Value type: {@link MetadataValueType# ARRAY}
    * Component type: {@link MetadataValueType# INT64}
    *
    * @param key
    *   the key to associate the array with
    * @param value
    *   the long array to set
    * @return
    *   this builder instance
    */
  def putArrayOfLong(key: String, value: Array[Long]): Builder

  /** Sets an unsigned long array for the given metadata key. Value type: {@link MetadataValueType#
    * ARRAY} Component type: {@link MetadataValueType# UINT64}
    *
    * @param key
    *   the key to associate the array with
    * @param value
    *   the unsigned long array to set
    * @return
    *   this builder instance
    */
  def putArrayOfUnsignedLong(key: String, value: Array[Long]): Builder

  /** Sets a float array for the given metadata key. Value type: {@link MetadataValueType# ARRAY}
    * Component type: {@link MetadataValueType# FLOAT32}
    *
    * @param key
    *   the key to associate the array with
    * @param value
    *   the float array to set
    * @return
    *   this builder instance
    */
  def putArrayOfFloat(key: String, value: Array[Float]): Builder

  /** Sets a double array for the given metadata key. Value type: {@link MetadataValueType# ARRAY}
    * Component type: {@link MetadataValueType# FLOAT64}
    *
    * @param key
    *   the key to associate the array with
    * @param value
    *   the double array to set
    * @return
    *   this builder instance
    */
  def putArrayOfDouble(key: String, value: Array[Double]): Builder
}
