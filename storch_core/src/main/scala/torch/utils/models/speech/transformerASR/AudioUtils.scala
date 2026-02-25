package torch.utils.models.speech.transformerASR

//import org.apache.commons.math3.transform.{DftNormalization, FastFourierTransformer, TransformType}
//import org.apache.commons.math3.util.FastMath
import torch.*
//import torch.nn.modules.{HasParams, TensorModule}
//import torch.nn.{Dropout, Embedding, Linear, Transformer, functional as F}

import java.io.File
import java.nio.file.Paths
import javax.sound.sampled.AudioSystem
import scala.collection.mutable.ArrayBuffer

// MFCC特征提取工具类
object AudioUtils:
  // 提取MFCC特征（参考librosa实现）
  def extractMFCC(
      audioPath: String,
      n_mfcc: Int = 13,
      sr: Int = 16000,
      n_fft: Int = 512,
      hopLength: Int = 160
  ): Tensor[Float32] =
    // 1. 加载音频文件（简化实现，实际需处理音频读取）
    val audioFile = new File(audioPath)
    val audioInputStream = AudioSystem.getAudioInputStream(audioFile)
    val format = audioInputStream.getFormat()
    val frameLength = audioInputStream.getFrameLength().toInt
    val buffer = new Array[Byte](frameLength * format.getFrameSize())
    audioInputStream.read(buffer)

    // 2. 音频信号转换为浮点数（简化处理，实际需考虑位深和通道数）
    val y = buffer
      .grouped(format.getFrameSize())
      .map(bytes => bytes(0).toFloat / 128.0f) // 假设单通道8位音频
      .toArray

    // 3. 计算MFCC特征（核心逻辑，实际需实现Mel频谱和DCT变换）
    val mfcc = computeMFCC(y, sr, n_mfcc, n_fft, hopLength)

    // 4. 计算一阶和二阶差分
    val deltaMFCC = computeDelta(mfcc)
    val delta2MFCC = computeDelta(deltaMFCC)

    // 5. 拼接特征并转置为 (时间步, 特征维度)
    val features =
      mfcc.zip(deltaMFCC).zip(delta2MFCC).map { case ((m, d1), d2) => m ++ d1 ++ d2 }.toArray

    Tensor(features.map(arr => arr.toSeq)) // 返回Tensor[Float32]

  // 简化的MFCC计算（实际需依赖频谱分析库）
  private def computeMFCC(
      y: Array[Float],
      sr: Int,
      n_mfcc: Int,
      n_fft: Int,
      hopLength: Int
  ): Array[Array[Float]] =
    // 此处省略Mel滤波器组和DCT变换细节，实际实现需参考 librosa.mfcc
    ArrayBuffer.fill(10)(Array.fill(n_mfcc)(0.0f)).toArray

  // 计算差分特征
  private def computeDelta(features: Array[Array[Float]], window: Int = 2): Array[Array[Float]] =
    val delta = ArrayBuffer[Array[Float]]()
    for (t <- features.indices) {
      val deltaT = ArrayBuffer[Float]()
      for (d <- features(t).indices) {
        var numerator = 0.0f
        var denominator = 0.0f
        for (k <- -window to window if k != 0) {
          val t_k = t + k
          if (t_k >= 0 && t_k < features.length) {
            numerator += k * features(t_k)(d)
            denominator += k * k
          }
        }
        deltaT += (if (denominator == 0) 0.0f else numerator / denominator)
      }
      delta += deltaT.toArray
    }
    delta.toArray
