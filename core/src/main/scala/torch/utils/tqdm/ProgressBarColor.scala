package torch
package utils.tqdm

import java.awt.Color

enum ProgressBarColor {
  case Pink, Green, Purple, Blue, Yellow, Cyan, White, Red, Orange, Gray, Black

}

object ProgressBarColor {
  val colorCodeMap = Map(
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

  def getColorUnicode(color: ProgressBarColor) = colorCodeMap.getOrElse(color, "\u001B[32m")

  def getSystemColor(color: ProgressBarColor) = color match
    case ProgressBarColor.Pink   => Color.PINK
    case ProgressBarColor.Green  => Color.GREEN
    case ProgressBarColor.Purple => new Color(128, 0, 128)
    case ProgressBarColor.Blue   => Color.BLUE
    case ProgressBarColor.Yellow => Color.YELLOW
    case ProgressBarColor.Cyan   => Color.CYAN
    case ProgressBarColor.White  => Color.WHITE
    case ProgressBarColor.Red    => Color.RED
    case ProgressBarColor.Orange => new Color(255, 165, 0)
    case ProgressBarColor.Gray   => Color.GRAY
    case ProgressBarColor.Black  => Color.BLACK
    case _                       => Color.BLUE

  def apply(color: String): ProgressBarColor = {
    color match {
      case "pink"   => ProgressBarColor.Pink
      case "green"  => ProgressBarColor.Green
      case "purple" => ProgressBarColor.Purple
      case "blue"   => ProgressBarColor.Blue
      case "yellow" => ProgressBarColor.Yellow
      case "cyan"   => ProgressBarColor.Cyan
      case "white"  => ProgressBarColor.White
      case "red"    => ProgressBarColor.Red
      case "orange" => ProgressBarColor.Orange
      case "gray"   => ProgressBarColor.Gray
      case "black"  => ProgressBarColor.Black
    }
  }
}
