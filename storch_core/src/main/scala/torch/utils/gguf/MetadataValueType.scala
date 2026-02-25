package torch.utils.gguf

/** Represents the different types of metadata values that can be stored in a GGUF file. Each type
  * has an associated byte size, which can be either positive for fixed-size types or negative to
  * indicate variable-length types with a length prefix.
  */

enum MetadataValueType:
  case UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64,
    FLOAT64

  def byteSize(): Int = this match
    case UINT8 | INT8 | BOOL      => 1
    case UINT16 | INT16           => 2
    case UINT32 | INT32 | FLOAT32 => 4
    case UINT64 | INT64 | FLOAT64 => 8
    case STRING | ARRAY           => -8

//enum MetadataValueType:
//  case UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64

object MetadataValueType:

  def fromIndex(index: Int): MetadataValueType = index match {
    case 0  => UINT8
    case 1  => INT8
    case 2  => UINT16
    case 3  => INT16
    case 4  => UINT32
    case 5  => INT32
    case 6  => FLOAT32
    case 7  => BOOL
    case 8  => STRING
    case 9  => ARRAY
    case 10 => UINT64
    case 11 => INT64
    case 12 => FLOAT64
    case _  => throw new ArrayIndexOutOfBoundsException("Invalid index for MetadataValueType")
  }

  def byteSize: Int = byteSize

  /** The value is a 8-bit unsigned integer.
    */
//  case INT8
//  case UINT16
//  case INT16
//  case UINT32
//  case INT32
//  case FLOAT32
//  case BOOL
//  case STRING
//  case ARRAY
//  case UINT64
//  case INT64
//  case FLOAT64

//object MetadataValueType extends Enumeration {
//  type MetadataValueType = Value
//  val
//
//  /**
//   * The value is a 8-bit unsigned integer.
//   */
//  UINT8,
//
//  /**
//   * The value is a 8-bit signed integer.
//   */
//  INT8,
//
//  /**
//   * The value is a 16-bit unsigned little-endian integer.
//   */
//  UINT16,
//
//  /**
//   * The value is a 16-bit signed little-endian integer.
//   */
//  INT16,
//
//  /**
//   * The value is a 32-bit unsigned little-endian integer.
//   */
//  UINT32,
//
//  /**
//   * The value is a 32-bit signed little-endian integer.
//   */
//  INT32,
//
//  /**
//   * The value is a 32-bit IEEE754 floating point number.
//   */
//  FLOAT32,
//
//  /**
//   * The value is a boolean.
//   * 1-byte value where 0 is false and 1 is true.
//   * Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
//   */
//  BOOL,
//
//  /**
//   * The value is a UTF-8 non-null-terminated string, with length prepended.
//   */
//  STRING,
//
//  /**
//   * The value is an array of other values, with the length and type prepended.
//   * Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
//   */
//  ARRAY,
//
//  /**
//   * The value is a 64-bit unsigned little-endian integer.
//   */
//  UINT64,
//
//  /**
//   * The value is a 64-bit signed little-endian integer.
//   */
//  INT64,
//
//  /**
//   * The value is a 64-bit IEEE754 floating point number.
//   */
//  FLOAT64 = Value
//  /**
//   * The size in bytes for this value type.
//   * Positive values indicate fixed-size types.
//   * Negative values indicate variable-length types with a length prefix of the absolute value size.
//   */
//  private val byteSize = 0
//
//  /**
//   * Constructs a new {@link MetadataValueType} with the specified byte size.
//   *
//   * @param byteSize the size in bytes for this type. Positive for fixed-size types,
//   *                 negative for variable-length types indicating the size of their length prefix
//   */
//  def this(byteSize: Int) {
//    this()
//    this.byteSize = byteSize
//  }
//
//  /**
//   * Cache of enum values to avoid creating new arrays on each call to {@link # values ( )}.
//   */
//  private val VALUES = values

/** Returns the MetadataValueType corresponding to the specified index. This method is more
  * efficient than valueOf() as it uses a cached array of values.
  *
  * @param index
  *   the index of the desired MetadataValueType
  * @return
  *   the MetadataValueType at the specified index
  * @throws ArrayIndexOutOfBoundsException
  *   if the index is out of range
  */

/** Returns the byte size of this value type.
  *
  * @return
  *   for fixed-size types, returns the positive size in bytes; for variable-length types, returns
  *   the negative size of their length prefix
  */
