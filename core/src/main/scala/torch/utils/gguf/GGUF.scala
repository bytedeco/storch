package torch.utils.gguf

//import gguf.impl.ImplAccessor

import torch.utils.gguf.impl.ImplAccessor

import java.io.IOException
import java.nio.channels.{ReadableByteChannel, WritableByteChannel}
import java.nio.file.{Files, Path, StandardOpenOption}
import java.util

/** Interface for handling GGUF files, which are used to store large language models and their
  * associated metadata. <p> This interface provides methods to read and write GGUF files, access
  * their metadata, and manage tensor information. GGUF is a binary format that includes both model
  * weights and associated configuration data.
  *
  * @see
  *   <a href="https://github.com/ggerganov/ggml/blob/master/docs/gguf.md">GGUF format
  *   specification</a>
  */
object GGUF {

  /** Reads GGUF metadata from a {@link ReadableByteChannel}.
    *
    * @param byteChannel
    *   the channel to read from
    * @return
    *   a new GGUF instance containing the metadata
    * @throws IOException
    *   if an I/O error occurs during reading
    */
  @throws[IOException]
  def read(byteChannel: ReadableByteChannel): GGUF = ImplAccessor.read(byteChannel)

  /** Reads a GGUF file from a path.
    *
    * @param modelPath
    *   the path to the GGUF file
    * @return
    *   a new GGUF instance containing the medatada
    * @throws IOException
    *   if an I/O error occurs during reading
    */
  @throws[IOException]
  def read(modelPath: Path): GGUF = try {
    val byteChannel = Files.newByteChannel(modelPath, StandardOpenOption.READ)
    try read(byteChannel)
    finally if (byteChannel != null) byteChannel.close()
  }

  /** Writes GGUF metadata to a {@link WritableByteChannel}.
    *
    * @param gguf
    *   the GGUF instance to write
    * @param byteChannel
    *   the channel to write to
    * @throws IOException
    *   if an I/O error occurs during writing
    */
  @throws[IOException]
  def write(gguf: GGUF, byteChannel: WritableByteChannel): Unit = {
    ImplAccessor.write(gguf, byteChannel)
  }

  /** Writes a GGUF instance to a file at the specified path.
    *
    * @param gguf
    *   the GGUF instance to write
    * @param modelPath
    *   the path where the GGUF file should be written
    * @throws IOException
    *   if an I/O error occurs during writing
    */
  @throws[IOException]
  def write(gguf: GGUF, modelPath: Path): Unit = {
    try {
      val byteChannel =
        Files.newByteChannel(modelPath, StandardOpenOption.WRITE, StandardOpenOption.CREATE_NEW)
      try write(gguf, byteChannel)
      finally if (byteChannel != null) byteChannel.close()
    }
  }
}

trait GGUF {

  /** Returns the version number of the GGUF instance.
    *
    * @return
    *   the GGUF format version as an integer
    */
  def getVersion: Int

  /** Returns the alignment value used in the GGUF file. If not explicitly set, returns the default
    * alignment value. <p> The alignment determines the byte alignment requirements for tensor data.
    *
    * @return
    *   the alignment value, or the default alignment if not specified
    */
  def getAlignment: Int = {
    if (containsKey(ImplAccessor.alignmentKey)) {
      assert(getType(ImplAccessor.alignmentKey) eq MetadataValueType.UINT32)
      return getValue(classOf[Int], ImplAccessor.alignmentKey)
    }
    ImplAccessor.defaultAlignment
  }

  /** Returns the byte offset where tensor data begins in the GGUF file.
    *
    * @return
    *   the byte offset to the start of tensor data
    */
  def getTensorDataOffset: Long

  /** Returns a set of all metadata keys present in the GGUF metadata, order is preserved.
    *
    * @return
    *   a set containing all metadata keys
    */
  def getMetadataKeys: Set[String]

  /** Gets a metadata value associated with the given key, casting it to the specified target class,
    * or null if the key is not found. The method handles primitive types, wrapper classes, strings,
    * and arrays.
    *
    * <p>The actual type of the stored value depends on {@link # getType ( String )}: <ul>
    * <li>{@code UINT8} → {@code byte} (signed, may require manual unsigned conversion) <li>{@code
    * INT8} → {@code byte} <li>{@code UINT16} → {@code short} (signed, may require manual unsigned
    * conversion) <li>{@code INT16} → {@code short} <li>{@code UINT32} → {@code int} (signed, may
    * require manual unsigned conversion) <li>{@code INT32} → {@code int} <li>{@code FLOAT32} →
    * {@code float} <li>{@code BOOL} → {@code boolean} <li>{@code STRING} → {@code String}
    * <li>{@code UINT64} → {@code long} (signed, may require manual unsigned conversion) <li>{@code
    * INT64} → {@code long} <li>{@code FLOAT64} → {@code double} <li>{@code ARRAY} → Array type
    * depends on {@link # getComponentType ( String )}: <ul> <li>{@code STRING} → {@code String[]}
    * <li>{@code UINT8} → {@code byte[]} (signed values) <li>{@code INT8} → {@code byte[]}
    * <li>{@code UINT16} → {@code short[]} (signed values) <li>{@code INT16} → {@code short[]}
    * <li>{@code UINT32} → {@code int[]} (signed values) <li>{@code INT32} → {@code int[]}
    * <li>{@code UINT64} → {@code long[]} (signed values) <li>{@code INT64} → {@code long[]}
    * <li>{@code FLOAT32} → {@code float[]} <li>{@code FLOAT64} → {@code double[]} <li>{@code BOOL}
    * → {@code boolean[]} </ul> </ul>
    *
    * <p>Examples: <pre>{@code // Primitive types // Will throw NullPointerException is the key is
    * not present. int intValue = getValue(int.class, "numberKey"); boolean flag =
    * getValue(boolean.class, "flagKey");
    *
    * // Wrapper classes Integer boxedInt = getValue(Integer.class, "numberKey"); Boolean boxedFlag
    * \= getValue(Boolean.class, "flagKey");
    *
    * // Strings String text = getValue(String.class, "textKey");
    *
    * // Arrays int[] numbers = getValue(int[].class, "numberArrayKey"); float[] floats =
    * getValue(float[].class, "floatArrayKey"); String[] strings = getValue(String[].class,
    * "stringArrayKey");
    *
    * // For generic access without type checking Object generic = getValue(Object.class, "anyKey");
    * }</pre>
    *
    * @param <T>
    *   the type to cast the value to
    * @param targetClass
    *   the Class object representing the desired return type
    * @param key
    *   the key whose associated value is to be returned
    * @return
    *   the value associated with the key, cast to type T, or null if the key is not found
    * @throws ClassCastException
    *   if the value cannot be cast to the requested type or if the requested type doesn't match the
    *   type indicated by {@link # getType ( String )}
    *
    * @see
    *   #getType(String)
    * @see
    *   #getComponentType(String)
    */
  def getValue[T](targetClass: Class[T], key: String): T

  /** Retrieves the value associated with the specified metadata key, or returns a default value if
    * the key is not present.
    *
    * @param <
    *   T> the expected type of the value
    * @param key
    *   the metadata key to look up
    * @param defaultValue
    *   the value to return if the key is not found
    * @return
    *   the value associated with the key, or defaultValue if not found
    * @see
    *   #getValue(Class, String)
    */
  def getValueOrDefault[T](targetClass: Class[T], key: String, defaultValue: T): T = if (
    containsKey(key)
  ) getValue(targetClass, key)
  else defaultValue

  /** Checks if a metadata key exists.
    *
    * @param key
    *   the metadata key to check
    * @return
    *   true if the key exists, false otherwise
    */
  def containsKey(key: String): Boolean = getValue(classOf[AnyRef], key) != null

  /** Returns the metadata value type for the specified key.
    *
    * @param key
    *   the metadata key to look up
    * @return
    *   the {@link MetadataValueType} of the value associated with the key, or null if not found
    */
  def getType(key: String): MetadataValueType

  /** Returns the component type for {@link MetadataValueType# ARRAY array} metadata values.
    *
    * @param key
    *   the metadata key to look up
    * @return
    *   the {@link MetadataValueType} of the array components, or null if not found or the value
    *   associated with the given key is not an {@link MetadataValueType# ARRAY array}
    */
  def getComponentType(key: String): MetadataValueType

  /** Returns information about all tensors stored in the GGUF metadata, order is preserved.
    *
    * @return
    *   a collection of {@link TensorInfo} objects describing all tensors
    */
  def getTensors: Seq[TensorInfo]

  /** Returns information about a specific tensor by name.
    *
    * @param tensorName
    *   the name of the tensor to look up
    * @return
    *   the {@link TensorInfo} for the specified tensor, or null if not found
    */
  def getTensor(tensorName: String): TensorInfo

  /** Checks if a tensor with the specified name exists in the GGUF file.
    *
    * @param tensorName
    *   the name of the tensor to check
    * @return
    *   true if the tensor exists, false otherwise
    */
  def containsTensor(tensorName: String): Boolean = getTensor(tensorName) != null
}
