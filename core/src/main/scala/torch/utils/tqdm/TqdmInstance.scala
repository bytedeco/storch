package torch
package utils.tqdm

object TqdmInstance {
  private[tqdm] val DEFAULT_NCOLS = 100

  def tqdm[T](a: List[T], color: Option[ProgressBarColor]) = if color.isDefined then
    new Tqdm[T](a.iterator, a.size, "", DEFAULT_NCOLS, false, color)
  else new Tqdm[T](a.iterator, a.size, "", DEFAULT_NCOLS, false, Some(ProgressBarColor.Green))

  def tqdm(total: Int, desc: String, color: Option[ProgressBarColor]) = if color.isDefined then
    new Tqdm[AnyRef](null, total, desc, DEFAULT_NCOLS, false, color)
  else new Tqdm[AnyRef](null, total, desc, DEFAULT_NCOLS, false, Some(ProgressBarColor.Green))

  def tqdm(total: Int, desc: String, gui: Boolean, color: Option[ProgressBarColor]) =
    if color.isDefined then new Tqdm[AnyRef](null, total, desc, DEFAULT_NCOLS, gui, color)
    else new Tqdm[AnyRef](null, total, desc, DEFAULT_NCOLS, gui, Some(ProgressBarColor.Purple))

  def tqdm[T](a: Iterator[T], total: Int, desc: String, color: Option[ProgressBarColor]) =
    if color.isDefined then new Tqdm[T](a, total, desc, DEFAULT_NCOLS, false, color)
    else new Tqdm[T](a, total, desc, DEFAULT_NCOLS, false, Some(ProgressBarColor.Green))

  def tqdm[T](a: List[T], desc: String, color: Option[ProgressBarColor]) = if color.isDefined then
    new Tqdm[T](a.iterator, a.size, desc, DEFAULT_NCOLS, false, color)
  else new Tqdm[T](a.iterator, a.size, desc, DEFAULT_NCOLS, false, Some(ProgressBarColor.Green))

  def tqdm[T](a: List[T], gui: Boolean, color: Option[ProgressBarColor]) = if color.isDefined then
    new Tqdm[T](a.iterator, a.size, "", DEFAULT_NCOLS, gui, color)
  else new Tqdm[T](a.iterator, a.size, "", DEFAULT_NCOLS, gui, Some(ProgressBarColor.Purple))

  def tqdm[T](a: List[T], width: Int, color: Option[ProgressBarColor]) = if color.isDefined then
    new Tqdm[T](a.iterator, a.size, "", DEFAULT_NCOLS, false, color)
  else new Tqdm[T](a.iterator, a.size, "", DEFAULT_NCOLS, false, Some(ProgressBarColor.Green))

  def tqdm[T](a: Seq[T], desc: String, color: Option[ProgressBarColor]) = if color.isDefined then
    new Tqdm[T](a.iterator, a.size, desc, DEFAULT_NCOLS, false, color)
  else new Tqdm[T](a.iterator, a.size, desc, DEFAULT_NCOLS, false, Some(ProgressBarColor.Green))

  def tqdm[T](
      a: Seq[T],
      gui: Boolean,
      color: Option[ProgressBarColor] = Some(ProgressBarColor.Purple)
  ) = if color.isDefined then new Tqdm[T](a.iterator, a.size, "", DEFAULT_NCOLS, gui, color)
  else new Tqdm[T](a.iterator, a.size, "", DEFAULT_NCOLS, gui, Some(ProgressBarColor.Purple))

  def tqdm[T](a: Seq[T], width: Int, color: Option[ProgressBarColor]) = if color.isDefined then
    new Tqdm[T](a.iterator, a.size, "", DEFAULT_NCOLS, false, color)
  else new Tqdm[T](a.iterator, a.size, "", DEFAULT_NCOLS, false, Some(ProgressBarColor.Green))

  def tqdm[T](a: Set[T], desc: String, color: Option[ProgressBarColor]) = if color.isDefined then
    new Tqdm[T](a.iterator, a.size, desc, DEFAULT_NCOLS, false, color)
  else new Tqdm[T](a.iterator, a.size, desc, DEFAULT_NCOLS, false, Some(ProgressBarColor.Green))

  def tqdm[T](a: Set[T], gui: Boolean, color: Option[ProgressBarColor]) = if color.isDefined then
    new Tqdm[T](a.iterator, a.size, "", DEFAULT_NCOLS, gui, color)
  else new Tqdm[T](a.iterator, a.size, "", DEFAULT_NCOLS, gui, Some(ProgressBarColor.Purple))

  def tqdm[T](a: Set[T], width: Int, color: Option[ProgressBarColor]) = if color.isDefined then
    new Tqdm[T](a.iterator, a.size, "", DEFAULT_NCOLS, false, color)
  else new Tqdm[T](a.iterator, a.size, "", DEFAULT_NCOLS, false, Some(ProgressBarColor.Green))

  def tqdm[T](a: Map[String, T], desc: String, color: Option[ProgressBarColor]) = if color.isDefined
  then new Tqdm[T](a.valuesIterator, a.size, desc, DEFAULT_NCOLS, false, color)
  else
    new Tqdm[T](a.valuesIterator, a.size, desc, DEFAULT_NCOLS, false, Some(ProgressBarColor.Green))

  def tqdm[T](a: Map[String, T], gui: Boolean, color: Option[ProgressBarColor]) = if color.isDefined
  then new Tqdm[T](a.valuesIterator, a.size, "", DEFAULT_NCOLS, gui, color)
  else new Tqdm[T](a.valuesIterator, a.size, "", DEFAULT_NCOLS, gui, Some(ProgressBarColor.Purple))

  def tqdm[T](a: Map[String, T], width: Int, color: Option[ProgressBarColor]) = if color.isDefined
  then new Tqdm[T](a.valuesIterator, a.size, "", DEFAULT_NCOLS, false, color)
  else new Tqdm[T](a.valuesIterator, a.size, "", DEFAULT_NCOLS, false, Some(ProgressBarColor.Green))
}
