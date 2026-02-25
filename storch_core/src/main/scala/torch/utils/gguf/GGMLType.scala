package torch.utils.gguf

import scala.collection.immutable.ArraySeq
import scala.collection.mutable

enum GGMLType(val blockByteSize: Int, val elementsPerBlock: Int, val isQuantized: Boolean):
  case F32(override val blockByteSize: Int) extends GGMLType(blockByteSize, 1, false)
  case F16(override val blockByteSize: Int) extends GGMLType(blockByteSize, 1, false)
  case Q4_0(override val blockByteSize: Int, override val elementsPerBlock: Int)
      extends GGMLType(blockByteSize, elementsPerBlock, true)
  case Q4_1(override val blockByteSize: Int, override val elementsPerBlock: Int)
      extends GGMLType(blockByteSize, elementsPerBlock, true)
  case Q4_2(override val blockByteSize: Int) extends GGMLType(blockByteSize, 1, false)
  case Q4_3(override val blockByteSize: Int) extends GGMLType(blockByteSize, 1, false)
  case Q5_0(override val blockByteSize: Int) extends GGMLType(blockByteSize, 1, false)
  case Q5_1(override val blockByteSize: Int) extends GGMLType(blockByteSize, 1, false)
  case Q8_0(override val blockByteSize: Int, override val elementsPerBlock: Int)
      extends GGMLType(blockByteSize, elementsPerBlock, true)
  case Q8_1(override val blockByteSize: Int, override val elementsPerBlock: Int)
      extends GGMLType(blockByteSize, elementsPerBlock, true)
  case Q2_K(override val blockByteSize: Int) extends GGMLType(blockByteSize, 1, false)
  case Q3_K(override val blockByteSize: Int) extends GGMLType(blockByteSize, 1, false)
  case Q4_K(override val blockByteSize: Int, override val elementsPerBlock: Int)
      extends GGMLType(blockByteSize, elementsPerBlock, true)
  case Q5_K(override val blockByteSize: Int, override val elementsPerBlock: Int)
      extends GGMLType(blockByteSize, elementsPerBlock, true)
  case Q6_K(override val blockByteSize: Int, override val elementsPerBlock: Int)
      extends GGMLType(blockByteSize, elementsPerBlock, true)
  case Q8_K(override val blockByteSize: Int) extends GGMLType(blockByteSize, 1, false)
  case I8(override val blockByteSize: Int) extends GGMLType(blockByteSize, 1, false)
  case I16(override val blockByteSize: Int) extends GGMLType(blockByteSize, 1, false)
  case I32(override val blockByteSize: Int) extends GGMLType(blockByteSize, 1, false)
  case I64(override val blockByteSize: Int) extends GGMLType(blockByteSize, 1, false)
  case F64(override val blockByteSize: Int) extends GGMLType(blockByteSize, 1, false)
  case IQ1_M(override val blockByteSize: Int) extends GGMLType(blockByteSize, 1, false)
  case BF16(override val blockByteSize: Int) extends GGMLType(blockByteSize, 1, false)
  case Q4_0_4_4(override val blockByteSize: Int, override val elementsPerBlock: Int)
      extends GGMLType(blockByteSize, elementsPerBlock, true)
  case Q4_0_4_8(override val blockByteSize: Int, override val elementsPerBlock: Int)
      extends GGMLType(blockByteSize, elementsPerBlock, true)
  case Q4_0_8_8(override val blockByteSize: Int, override val elementsPerBlock: Int)
      extends GGMLType(blockByteSize, elementsPerBlock, true)
  case TQ1_0(override val blockByteSize: Int) extends GGMLType(blockByteSize, 1, false)
  case TQ2_0(override val blockByteSize: Int) extends GGMLType(blockByteSize, 1, false)

  def getBlockByteSize(): Int = blockByteSize
  def getElementsPerBlock(): Int = elementsPerBlock
  def getIsQuantized(): Boolean = isQuantized

  def byteSizeForShape(dims: Long*): Long = byteSizeFor(numberOfElements(dims*))

  private def numberOfElements(dims: Long*): Long =
    assert(dims.forall(_ > 0))
    dims.product

  def byteSizeFor(numberOfElements: Long): Long =
    val t = numberOfElements * getBlockByteSize()
    assert(t % getElementsPerBlock() == 0)
    t / getElementsPerBlock()

object GGMLType:
  private val FLOAT16_BYTES = 2
  private val BFLOAT16_BYTES = 2
//  private val VALUES = ArraySeq(GGMLType.values*)

  private val QK_K = 256 // or 64?
  private val QK4_0 = 32
  private val QK8_0 = 32

  def fromId(id: Int): GGMLType = id match
    case 0  => GGMLType.F32(java.lang.Float.BYTES)
    case 1  => GGMLType.F16(GGMLType.FLOAT16_BYTES)
    case 2  => GGMLType.Q4_0(GGMLType.FLOAT16_BYTES + 16 * java.lang.Byte.BYTES, GGMLType.QK4_0)
    case 3  => GGMLType.Q4_1(2 * GGMLType.FLOAT16_BYTES + 16 * java.lang.Byte.BYTES, GGMLType.QK4_0)
    case 4  => GGMLType.Q4_2(Int.MaxValue)
    case 5  => GGMLType.Q4_3(Int.MaxValue)
    case 6  => GGMLType.Q5_0(Int.MaxValue)
    case 7  => GGMLType.Q5_1(Int.MaxValue)
    case 8  => GGMLType.Q8_0(GGMLType.FLOAT16_BYTES + 32 * java.lang.Byte.BYTES, GGMLType.QK8_0)
    case 9  => GGMLType.Q8_1(32 * java.lang.Byte.BYTES + 2 * java.lang.Float.BYTES, GGMLType.QK8_0)
    case 10 => GGMLType.Q2_K(Int.MaxValue)
    case 11 => GGMLType.Q3_K(Int.MaxValue)
    case 12 =>
      GGMLType.Q4_K(
        2 * GGMLType.FLOAT16_BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 2,
        GGMLType.QK_K
      )
    case 13 =>
      GGMLType.Q5_K(
        2 * GGMLType.FLOAT16_BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 8 + GGMLType.QK_K / 2,
        GGMLType.QK_K
      )
    case 14 =>
      GGMLType.Q6_K(
        GGMLType.QK_K / 2 + GGMLType.QK_K / 4 + GGMLType.QK_K / 16 + GGMLType.FLOAT16_BYTES,
        GGMLType.QK_K
      )
    case 15 => GGMLType.Q8_K(Int.MaxValue)
    case 16 => GGMLType.I8(java.lang.Byte.BYTES)
    case 17 => GGMLType.I16(java.lang.Short.BYTES)
    case 18 => GGMLType.I32(java.lang.Integer.BYTES)
    case 19 => GGMLType.I64(java.lang.Long.BYTES)
    case 20 => GGMLType.F64(java.lang.Double.BYTES)
    case 21 => GGMLType.IQ1_M(Int.MaxValue)
    case 22 => GGMLType.BF16(GGMLType.BFLOAT16_BYTES)
    case 23 => GGMLType.Q4_0_4_4(GGMLType.FLOAT16_BYTES + 16 * java.lang.Byte.BYTES, GGMLType.QK4_0)
    case 24 => GGMLType.Q4_0_4_8(GGMLType.FLOAT16_BYTES + 16 * java.lang.Byte.BYTES, GGMLType.QK4_0)
    case 25 => GGMLType.Q4_0_8_8(GGMLType.FLOAT16_BYTES + 16 * java.lang.Byte.BYTES, GGMLType.QK4_0)
    case 26 => GGMLType.TQ1_0(Int.MaxValue)
    case 27 => GGMLType.TQ2_0(Int.MaxValue)

  def apply(blockByteSize: Int): GGMLType =
    // 示例逻辑，根据 blockByteSize 匹配预定义的枚举实例
    blockByteSize match
      case java.lang.Float.BYTES   => F32(blockByteSize)
      case `FLOAT16_BYTES`         => F16(blockByteSize)
      case `BFLOAT16_BYTES`        => BF16(blockByteSize)
      case java.lang.Byte.BYTES    => I8(blockByteSize)
      case java.lang.Short.BYTES   => I16(blockByteSize)
      case java.lang.Integer.BYTES => I32(blockByteSize)
      case java.lang.Long.BYTES    => I64(blockByteSize)
      case java.lang.Double.BYTES  => F64(blockByteSize)
      case _ =>
        throw new IllegalArgumentException(s"No GGMLType found for blockByteSize: $blockByteSize")

  def apply(blockByteSize: Int, elementsPerBlock: Int): GGMLType =
    // 示例逻辑，根据 blockByteSize 和 elementsPerBlock 匹配预定义的枚举实例
    if elementsPerBlock == QK4_0 && blockByteSize == FLOAT16_BYTES + 16 * java.lang.Byte.BYTES then
      Q4_0(blockByteSize, elementsPerBlock)
    else if elementsPerBlock == QK4_0 && blockByteSize == 2 * FLOAT16_BYTES + 16 * java.lang.Byte.BYTES
    then Q4_1(blockByteSize, elementsPerBlock)
    else if elementsPerBlock == QK8_0 && blockByteSize == FLOAT16_BYTES + 32 * java.lang.Byte.BYTES
    then Q8_0(blockByteSize, elementsPerBlock)
    else if elementsPerBlock == QK8_0 && blockByteSize == 32 * java.lang.Byte.BYTES + 2 * java.lang.Float.BYTES
    then Q8_1(blockByteSize, elementsPerBlock)
    else if elementsPerBlock == QK_K && blockByteSize == 2 * FLOAT16_BYTES + ((QK_K / 16) / 8 * 6) + QK_K / 2
    then Q4_K(blockByteSize, elementsPerBlock)
    else
      throw new IllegalArgumentException(
        s"No GGMLType found for blockByteSize: $blockByteSize and elementsPerBlock: $elementsPerBlock"
      )

//  def apply(blockByteSize: Int): GGMLType =  GGMLType(blockByteSize, 1, false)
//  def apply(blockByteSize: Int, elementsPerBlock: Int): GGMLType =
//    new GGMLType(blockByteSize, elementsPerBlock, true)

  private def isPowerOf2(n: Int): Boolean = n > 0 && (n & (n - 1)) == 0
