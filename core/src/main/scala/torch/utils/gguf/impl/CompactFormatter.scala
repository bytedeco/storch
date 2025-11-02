package torch.utils.gguf.impl

import torch.utils.gguf.MetadataValueType.*

import java.lang.reflect.Array as JArray

/** Utility class for creating compact string representations of values. Useful for logging,
  * debugging, and display purposes where space is limited.
  */
object CompactFormatter {
  private val DEFAULT_MAX_LENGTH = 64
  private val DEFAULT_LIST_ITEMS = 4
  private val ELLIPSIS = "..."

  /** Creates a compact string representation of an object.
    *
    * @param obj
    *   the object to format
    * @return
    *   a compact string representation
    */
  def format(obj: AnyRef): String = format(obj, DEFAULT_MAX_LENGTH, DEFAULT_LIST_ITEMS)

  /** Creates a compact string representation of an object with custom limits.
    *
    * @param obj
    *   the object to format
    * @param maxLength
    *   maximum length for strings
    * @param maxItems
    *   maximum number of items to show in collections
    * @return
    *   a compact string representation
    */
  def format(obj: AnyRef, maxLength: Int, maxItems: Int): String = {
    obj match
      case null          => "null"
      case arr: Array[?] => formatArray(arr, maxItems)
      case str: String   => formatString(str, maxLength, addQuotes = true)
      case _             => formatString(obj.toString, maxLength, addQuotes = false)
//    if (obj == null) return "null"
//    // Handle arrays
//    if (obj.getClass.isArray) return formatArray(obj, maxItems)
//    // Handle strings
//    if (obj.isInstanceOf[String]) return formatString(obj.asInstanceOf[String], maxLength, true)
//    // For all other types, use toString() and truncate if needed
//    formatString(obj.toString, maxLength, false)
  }

  private def formatString(str: String, maxLength: Int, addQuotes: Boolean) = {
    var value = formatString2(str, maxLength)
    if (addQuotes) value = "\"" + value + "\""
    value
  }

  private def formatString2(str: String, maxLength: Int): String = {
    if (str.length <= maxLength) return str
    // Reserve space for ellipsis and quotes if present
    val targetLength = maxLength - ELLIPSIS.length
    str.substring(0, targetLength) + ELLIPSIS
  }

  // formatArray(array: Array[?], maxItems: Int)
  private def formatArray(array: AnyRef, maxItems: Int): String = {
    val length = JArray.getLength(array)
    if (length == 0) return "[]"
    val sb = new StringBuilder("[")
    val itemsToShow = Math.min(length, maxItems)
    for (i <- 0 until itemsToShow) {
      if (i > 0) sb.append(", ")
      val item = JArray.get(array, i)
      sb.append(format(item, DEFAULT_MAX_LENGTH, maxItems))
    }
    if (length > maxItems) sb.append(", ").append(ELLIPSIS)
    sb.append("]").toString
  }
}
