package torch
package utils.tqdm

import java.awt.{BorderLayout, Color, Font}
import javax.swing.{JFrame, JLabel, JProgressBar, WindowConstants}

class GuiProgress(desc: String, total: Int, color: ProgressBarColor = ProgressBarColor.Blue)
    extends JFrame {
  val widths = 500
  val heights = 200
  val colorPre = this.color match {
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
  }
  private[tqdm] var progress: JProgressBar = new JProgressBar
  this.setSize(widths, heights)
  this.setTitle(desc)
  this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
  progress.setMaximum(total)
  private[tqdm] var label: JLabel = new JLabel
  this.progress.setForeground(colorPre)
  label.setBackground(Color.red)
  label.setHorizontalTextPosition(0) // JLabel.CENTER)
  label.setFont(new Font("微软雅黑", Font.ITALIC, 25))
  this.setLayout(new BorderLayout)
  this.setLocationRelativeTo(null)
  this.add(progress, BorderLayout.CENTER)
  this.add(label, BorderLayout.SOUTH)
  this.setResizable(false)
  this.setVisible(true)

  private[tqdm] def complete(): Unit = {
    this.setVisible(true)
    try Thread.sleep(2000)
    catch {
      case e: InterruptedException =>
        e.printStackTrace()
    }
    this.dispose()
  }
}
