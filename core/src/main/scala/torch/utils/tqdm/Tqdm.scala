package torch
package utils.tqdm

import scala.util.Random

object Tqdm {
  private[tqdm] val DEFAULT_NCOLS = 100

  def tqdm[T](
      a: List[T],
      color: Option[ProgressBarColor],
      sleepSpeed: Option[Int],
      colorRandom: Boolean
  ) = if color.isDefined then
    new Tqdm[T](a.iterator, a.size, "", DEFAULT_NCOLS, false, color, sleepSpeed, colorRandom)
  else
    new Tqdm[T](
      a.iterator,
      a.size,
      "",
      DEFAULT_NCOLS,
      false,
      Some(ProgressBarColor.Green),
      sleepSpeed,
      colorRandom
    )

  def tqdm(
      total: Int,
      desc: String,
      color: Option[ProgressBarColor],
      sleepSpeed: Option[Int],
      colorRandom: Boolean
  ) = if color.isDefined then
    new Tqdm[AnyRef](null, total, desc, DEFAULT_NCOLS, false, color, sleepSpeed, colorRandom)
  else
    new Tqdm[AnyRef](
      null,
      total,
      desc,
      DEFAULT_NCOLS,
      false,
      Some(ProgressBarColor.Green),
      sleepSpeed,
      colorRandom
    )

  def tqdm(
      total: Int,
      desc: String,
      gui: Boolean,
      color: Option[ProgressBarColor],
      sleepSpeed: Option[Int],
      colorRandom: Boolean
  ) = if color.isDefined then
    new Tqdm[AnyRef](null, total, desc, DEFAULT_NCOLS, gui, color, sleepSpeed, colorRandom)
  else
    new Tqdm[AnyRef](
      null,
      total,
      desc,
      DEFAULT_NCOLS,
      gui,
      Some(ProgressBarColor.Purple),
      sleepSpeed,
      colorRandom
    )

  def tqdm[T](
      a: Iterator[T],
      total: Int,
      desc: String,
      color: Option[ProgressBarColor],
      sleepSpeed: Option[Int],
      colorRandom: Boolean
  ) = if color.isDefined then
    new Tqdm[T](a, total, desc, DEFAULT_NCOLS, false, color, sleepSpeed, colorRandom)
  else
    new Tqdm[T](
      a,
      total,
      desc,
      DEFAULT_NCOLS,
      false,
      Some(ProgressBarColor.Green),
      sleepSpeed,
      colorRandom
    )

  def tqdm[T](
      a: List[T],
      desc: String,
      color: Option[ProgressBarColor],
      sleepSpeed: Option[Int],
      colorRandom: Boolean
  ) = if color.isDefined then
    new Tqdm[T](a.iterator, a.size, desc, DEFAULT_NCOLS, false, color, sleepSpeed, colorRandom)
  else
    new Tqdm[T](
      a.iterator,
      a.size,
      desc,
      DEFAULT_NCOLS,
      false,
      Some(ProgressBarColor.Green),
      sleepSpeed,
      colorRandom
    )

  def tqdm[T](
      a: List[T],
      gui: Boolean,
      color: Option[ProgressBarColor],
      sleepSpeed: Option[Int],
      colorRandom: Boolean
  ) = if color.isDefined then
    new Tqdm[T](a.iterator, a.size, "", DEFAULT_NCOLS, gui, color, sleepSpeed, colorRandom)
  else
    new Tqdm[T](
      a.iterator,
      a.size,
      "",
      DEFAULT_NCOLS,
      gui,
      Some(ProgressBarColor.Purple),
      sleepSpeed,
      colorRandom
    )

  def tqdm[T](
      a: List[T],
      width: Int,
      color: Option[ProgressBarColor],
      sleepSpeed: Option[Int],
      colorRandom: Boolean
  ) = if color.isDefined then
    new Tqdm[T](a.iterator, a.size, "", DEFAULT_NCOLS, false, color, sleepSpeed, colorRandom)
  else
    new Tqdm[T](
      a.iterator,
      a.size,
      "",
      DEFAULT_NCOLS,
      false,
      Some(ProgressBarColor.Green),
      sleepSpeed,
      colorRandom
    )

  def tqdm[T](
      a: Seq[T],
      desc: String,
      color: Option[ProgressBarColor],
      sleepSpeed: Option[Int],
      colorRandom: Boolean
  ) = if color.isDefined then
    new Tqdm[T](a.iterator, a.size, desc, DEFAULT_NCOLS, false, color, sleepSpeed, colorRandom)
  else
    new Tqdm[T](
      a.iterator,
      a.size,
      desc,
      DEFAULT_NCOLS,
      false,
      Some(ProgressBarColor.Green),
      sleepSpeed,
      colorRandom
    )

  def tqdm[T](
      a: Seq[T],
      gui: Boolean,
      color: Option[ProgressBarColor] = Some(ProgressBarColor.Purple),
      sleepSpeed: Option[Int] = Some(100),
      colorRandom: Boolean = true
  ) = if color.isDefined then
    new Tqdm[T](a.iterator, a.size, "", DEFAULT_NCOLS, gui, color, sleepSpeed, colorRandom)
  else
    new Tqdm[T](
      a.iterator,
      a.size,
      "",
      DEFAULT_NCOLS,
      gui,
      Some(ProgressBarColor.Purple),
      sleepSpeed,
      colorRandom
    )

  def tqdm[T](
      a: Seq[T],
      width: Int,
      color: Option[ProgressBarColor],
      sleepSpeed: Option[Int],
      colorRandom: Boolean
  ) = if color.isDefined then
    new Tqdm[T](a.iterator, a.size, "", DEFAULT_NCOLS, false, color, sleepSpeed, colorRandom)
  else
    new Tqdm[T](
      a.iterator,
      a.size,
      "",
      DEFAULT_NCOLS,
      false,
      Some(ProgressBarColor.Green),
      sleepSpeed,
      colorRandom
    )

  def tqdm[T](
      a: Set[T],
      desc: String,
      color: Option[ProgressBarColor],
      sleepSpeed: Option[Int],
      colorRandom: Boolean
  ) = if color.isDefined then
    new Tqdm[T](a.iterator, a.size, desc, DEFAULT_NCOLS, false, color, sleepSpeed, colorRandom)
  else
    new Tqdm[T](
      a.iterator,
      a.size,
      desc,
      DEFAULT_NCOLS,
      false,
      Some(ProgressBarColor.Green),
      sleepSpeed,
      colorRandom
    )

  def tqdm[T](
      a: Set[T],
      gui: Boolean,
      color: Option[ProgressBarColor],
      sleepSpeed: Option[Int],
      colorRandom: Boolean
  ) = if color.isDefined then
    new Tqdm[T](a.iterator, a.size, "", DEFAULT_NCOLS, gui, color, sleepSpeed, colorRandom)
  else
    new Tqdm[T](
      a.iterator,
      a.size,
      "",
      DEFAULT_NCOLS,
      gui,
      Some(ProgressBarColor.Purple),
      sleepSpeed,
      colorRandom
    )

  def tqdm[T](
      a: Set[T],
      width: Int,
      color: Option[ProgressBarColor],
      sleepSpeed: Option[Int],
      colorRandom: Boolean
  ) = if color.isDefined then
    new Tqdm[T](a.iterator, a.size, "", DEFAULT_NCOLS, false, color, sleepSpeed, colorRandom)
  else
    new Tqdm[T](
      a.iterator,
      a.size,
      "",
      DEFAULT_NCOLS,
      false,
      Some(ProgressBarColor.Green),
      sleepSpeed,
      colorRandom
    )

  def tqdm[T](
      a: Map[String, T],
      desc: String,
      color: Option[ProgressBarColor],
      sleepSpeed: Option[Int],
      colorRandom: Boolean
  ) = if color.isDefined then
    new Tqdm[T](
      a.valuesIterator,
      a.size,
      desc,
      DEFAULT_NCOLS,
      false,
      color,
      sleepSpeed,
      colorRandom
    )
  else
    new Tqdm[T](
      a.valuesIterator,
      a.size,
      desc,
      DEFAULT_NCOLS,
      false,
      Some(ProgressBarColor.Green),
      sleepSpeed,
      colorRandom
    )

  def tqdm[T](
      a: Map[String, T],
      gui: Boolean,
      color: Option[ProgressBarColor],
      sleepSpeed: Option[Int],
      colorRandom: Boolean
  ) = if color.isDefined then
    new Tqdm[T](a.valuesIterator, a.size, "", DEFAULT_NCOLS, gui, color, sleepSpeed, colorRandom)
  else
    new Tqdm[T](
      a.valuesIterator,
      a.size,
      "",
      DEFAULT_NCOLS,
      gui,
      Some(ProgressBarColor.Purple),
      sleepSpeed,
      colorRandom
    )

  def tqdm[T](
      a: Map[String, T],
      width: Int,
      color: Option[ProgressBarColor],
      sleepSpeed: Option[Int],
      colorRandom: Boolean
  ) = if color.isDefined then
    new Tqdm[T](a.valuesIterator, a.size, "", DEFAULT_NCOLS, false, color, sleepSpeed, colorRandom)
  else
    new Tqdm[T](
      a.valuesIterator,
      a.size,
      "",
      DEFAULT_NCOLS,
      false,
      Some(ProgressBarColor.Green),
      sleepSpeed,
      colorRandom
    )
}

class Tqdm[T](
    var data: Iterator[T],
    var total: Int,
    var desc: String,
    var ncols: Int,
    var gui: Boolean,
    var color: Option[ProgressBarColor] = Some(ProgressBarColor.Purple),
    var sleepSpeed: Option[Int] = Some(50),
    var colorRandom: Boolean = true
) extends Iterable[T] {
//  private[tqdm] var ncols = 0
  val RED = "\u001B[31m"
  val GREEN = "\u001B[32m"
  val YELLOW = "\u001B[33m"
  val RESET = "\u001B[0m"
  private[tqdm] val printIntervalInMilli =
    if !sleepSpeed.isDefined || sleepSpeed.get <= 50 then 50 else sleepSpeed.get
  private[tqdm] val colorCodes = Map(
    ProgressBarColor.Pink -> "\u001B[38;5;206m",
    ProgressBarColor.Green -> "\u001B[32m",
    ProgressBarColor.Purple -> "\u001B[35m",
    ProgressBarColor.Blue -> "\u001B[34m",
    ProgressBarColor.Yellow -> "\u001B[33m",
    ProgressBarColor.Cyan -> "\u001B[36m",
    ProgressBarColor.White -> "\u001B[37m",
    ProgressBarColor.Red -> "\u001B[31m",
    ProgressBarColor.Orange -> "\u001B[38;5;208m",
    ProgressBarColor.Gray -> "\u001B[38;5;240m",
    ProgressBarColor.Black -> "\u001B[30m"
  )
  colorCodes.values.toSeq
  // State
  this.ncols =
    if (ncols == -1) Tqdm.DEFAULT_NCOLS
    else ncols
  private[tqdm] var lastPrintTime: Long = 0
  private[tqdm] var lastProgress = 0L
  private[tqdm] var beginTime = 0L
  private[tqdm] val unitChar = '█'
  private[tqdm] var chars: Array[Char] = new Array[Char](ncols)
  private[tqdm] var progress = 0 // -1
  this.beginTime = System.currentTimeMillis
  this.lastPrintTime = System.currentTimeMillis
  private[tqdm] var guiProgress: GuiProgress = null
//  this.chars = new Array[Char](ncols)
  if (gui) this.guiProgress = new GuiProgress(desc, total, color.get)

  def colorRandom(colorSeq: Seq[String]): String = Random.shuffle(colorSeq).head
  override def iterator: Iterator[T] = new Iterator[T]() {
    override def hasNext: Boolean = {
      if (data == null) throw new RuntimeException("tqdm need set data")
      data.hasNext
    }

    override def next: T = {
      update(1)
      //      Thread.sleep(1000)
      data.next
    }
  }

  def update(delta: Int): Unit = {
    progress += delta
//    println(s"update now first progress ${progress} ")
    if !sleepSpeed.isDefined || sleepSpeed.get <= 50 then Thread.sleep(50)
    else Thread.sleep(sleepSpeed.get)
    if (System.currentTimeMillis - lastPrintTime > printIntervalInMilli) {
      val speedString =
        formatSpeed(progress, lastProgress, System.currentTimeMillis - lastPrintTime)
//      println(s"update-> delta ${delta} lastProgress ${lastProgress} progress ${progress} total ${total} ")
      val percent = 1.0 * progress / total * 100
      val percentString = String.format("%2d%%", percent.toInt)
      val usedTime = System.currentTimeMillis - beginTime
      val leftTime = (usedTime * 1.0 / progress * (total - progress)).toLong
      val timeString =
        String.format("[%s<%s,%s]", formatTime(usedTime), formatTime(leftTime), speedString)
      if (gui) {
        this.guiProgress.progress.setValue(progress)
        this.guiProgress.label
          .setText(String.format("%s %d/%d %s", percentString, progress, total, timeString))
        if ((data != null && !data.hasNext) || this.progress >= this.total)
          this.guiProgress.complete()
      } else {
        val builder = new StringBuilder("\r")
        if (desc != null) builder.append(desc).append(':')
        // Percent String
        builder.append(percentString)
        // GuiProgress String
        val charCount = Math.min(progress * ncols / total, chars.length)
        java.util.Arrays.fill(chars, 0, charCount, unitChar)
        java.util.Arrays.fill(chars, charCount, ncols, ' ')
        builder.append('|').append(chars.mkString("")).append('|').append("\uD83C\uDF96\uFE0F")
        // GuiProgress Desc String
        builder.append(String.format("%d/%d", progress, total))
        builder.append(timeString)
        if colorRandom then
          print(s"${colorRandom(colorCodes.values.toSeq)}${builder.toString} ${RESET}")
        else print(s"${colorCodes.getOrElse(this.color.get, GREEN)}${builder.toString} ${RESET}")
        //        print(s"${GREEN}${builder.toString}${RESET}")
      }
      lastPrintTime = System.currentTimeMillis
      lastProgress = progress
    }
  }

  private[tqdm] def formatTime(durations: Long) = {
    var duration = durations
    val SECOND = 1000
    val MINUTE = 60 * SECOND
    val HOUR = 60 * MINUTE
    val DAY: Int = 24 * HOUR
    val day = duration / DAY
    duration = duration.toInt % DAY
    val hour = duration / HOUR
    duration %= HOUR
    val minute = duration / MINUTE
    duration %= MINUTE
    val second = duration / SECOND
    val builder = new StringBuilder
    if (day > 0) builder.append(String.format("%dd", day))
    if (hour > 0) builder.append(String.format("%dH", hour))
    if (minute > 0) builder.append(String.format("%dm", minute))
    if (second > 0) builder.append(String.format("%ds", second))
    builder.toString
  }

  private[tqdm] def formatSpeed(currentProgress: Long, lastProgress: Long, duration: Long) = {
    val progress = currentProgress - lastProgress
    if (progress == 1) String.format("%s/iter " + "\uD83C\uDF96\uFE0F", formatTime(duration))
    else String.format("%s iter/s " + "\uD83C\uDF96\uFE0F" + "\uD83C\uDF4E \uD83D\uDE80", progress)
  }
}
