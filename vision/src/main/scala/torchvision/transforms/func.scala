//package torchvision.transforms
//
//import org.bytedeco.pytorch.global.torch as torchGlobal
//import org.bytedeco.pytorch.*
//import java.util.EnumSet
//
//// 定义插值模式枚举
//object InterpolationMode extends Enumeration {
//  type InterpolationMode = Value
//  val NEAREST, NEAREST_EXACT, BILINEAR, BICUBIC, BOX, HAMMING, LANCZOS = Value
//}
//
//import InterpolationMode._
//
//object Functional {
//
//  val pilModesMapping: Map[InterpolationMode, Int] = Map(
//    NEAREST -> 0,
//    BILINEAR -> 2,
//    BICUBIC -> 3,
//    NEAREST_EXACT -> 0,
//    BOX -> 4,
//    HAMMING -> 5,
//    LANCZOS -> 1
//  )
//
//  def _interpolationModesFromInt(i: Int): InterpolationMode = {
//    i match {
//      case 0 => NEAREST
//      case 2 => BILINEAR
//      case 3 => BICUBIC
//      case 4 => BOX
//      case 5 => HAMMING
//      case 1 => LANCZOS
//      case _ => throw new IllegalArgumentException(s"Unsupported interpolation mode integer: $i")
//    }
//  }
//
//  def getDimensions(img: Tensor): Array[Int] = {
//    val sizes = img.sizes()
//    if (sizes.length >= 3) {
//      Array(sizes(sizes.length - 3), sizes(sizes.length - 2), sizes(sizes.length - 1))
//    } else {
//      throw new IllegalArgumentException("Input tensor does not have enough dimensions")
//    }
//  }
//
//  def getImageSize(img: Tensor): Array[Int] = {
//    val sizes = img.sizes()
//    if (sizes.length >= 2) {
//      Array(sizes(sizes.length - 1), sizes(sizes.length - 2))
//    } else {
//      throw new IllegalArgumentException("Input tensor does not have enough dimensions")
//    }
//  }
//
//  // 辅助函数，获取图像尺寸
//  def getImageNumChannels(img: Tensor): Int = {
//    val sizes = img.sizes()
//    if (sizes.length >= 3) {
//      sizes(sizes.length - 3)
//    } else {
//      throw new IllegalArgumentException("Input tensor does not have enough dimensions")
//    }
//  }
//
//
//  def _isNumpyImage(img: Any): Boolean = {
//    img.isInstanceOf[Tensor] && img.asInstanceOf[Tensor].sizes().length >= 2
//  }
//
//  def toTensor(pic: Tensor): Tensor = {
//    // 假设 pic 已经是合适的格式，可能需要更多处理
//    pic
//  }
//
//  def pilToTensor(pic: Tensor): Tensor = {
//    // 假设 pic 已经是合适的格式，可能需要更多处理
//    pic.clone()
//  }
//
//  def convertImageDtype(image: Tensor, dtype: ScalarType): Tensor = {
//    image.to(dtype)
//  }
//
//  def normalize(tensor: Tensor, mean: Array[Float], std: Array[Float], inplace: Boolean = false): Tensor = {
//    val result = if (inplace) tensor else tensor.clone()
//    val channels = getImageNumChannels(result)
//    for (i <- 0 until channels) {
//      result.index({
//        0
//      }, i).sub_(mean(i)).div_(std(i))
//    }
//    result
//  }
//
//  def _computeResizedOutputSize(
//                                 imageSize: (Int, Int),
//                                 size: Option[Array[Int]],
//                                 maxSize: Option[Int] = None,
//                                 allowSizeNone: Boolean = false
//                               ): Array[Int] = {
//    size match {
//      case Some(s) if s.length == 2 => s
//      case Some(s) if s.length == 1 =>
//        val (h, w) = imageSize
//        if (h > w) {
//          Array((s(0) * h / w).toInt, s(0))
//        } else {
//          Array(s(0), (s(0) * w / h).toInt)
//        }
//      case None if allowSizeNone => imageSize.productIterator.toArray.map(_.asInstanceOf[Int])
//      case _ => throw new IllegalArgumentException("Invalid size parameter")
//    }
//  }
//
//  def resize(
//              img: Tensor,
//              size: Array[Int],
//              interpolation: InterpolationMode = BILINEAR,
//              maxSize: Option[Int] = None,
//              antialias: Option[Boolean] = Some(true)
//            ): Tensor = {
//    val newSize = _computeResizedOutputSize((getImageSize(img)(1), getImageSize(img)(0)), Some(size), maxSize)
//    val mode = pilModesMapping(interpolation)
//    torchGlobal.nn.functional.interpolate(img.unsqueeze(0), newSize, mode = mode.toString.toLowerCase).squeeze(0)
//  }
//
//  def pad(
//           img: Tensor,
//           padding: Array[Int],
//           fill: Float = 0,
//           paddingMode: String = "constant"
//         ): Tensor = {
//    val padSizes = new Array[Int](padding.length * 2)
//    for (i <- padding.indices) {
//      padSizes(2 * i) = padding(i)
//      padSizes(2 * i + 1) = padding(i)
//    }
//    torchGlobal.nn.functional.pad(img, padSizes, paddingMode, fill)
//  }
//
//
//  // 辅助函数，裁剪图像
//  def crop(img: Tensor, top: Int, left: Int, height: Int, width: Int): Tensor = {
//    img.slice(-2, top, top + height).slice(-1, left, left + width)
//  }
//
//  def centerCrop(
//                  img: Tensor,
//                  outputSize: Array[Int]
//                ): Tensor = {
//    val (imgHeight, imgWidth) = (getImageSize(img)(1), getImageSize(img)(0))
//    val top = (imgHeight - outputSize(0)) / 2
//    val left = (imgWidth - outputSize(1)) / 2
//    crop(img, top, left, outputSize(0), outputSize(1))
//  }
//
//  def resizedCrop(
//                   img: Tensor,
//                   top: Int,
//                   left: Int,
//                   height: Int,
//                   width: Int,
//                   size: Array[Int],
//                   interpolation: InterpolationMode = BILINEAR,
//                   antialias: Option[Boolean] = Some(true)
//                 ): Tensor = {
//    val croppedImg = crop(img, top, left, height, width)
//    resize(croppedImg, size, interpolation, antialias = antialias)
//  }
//
//
//  // 辅助函数，水平翻转图像
//  def hflip(img: Tensor): Tensor = {
//    torchGlobal.flip(img, -1)
//  }
//
//  // 辅助函数，垂直翻转图像
//  def vflip(img: Tensor): Tensor = {
//    torchGlobal.flip(img, -2)
//  }
//
//  // 辅助函数，中心裁剪图像
////  def centerCrop(
////                  img: Tensor,
////                  outputSize: Array[Int]
////                ): Tensor = {
////    val (imgHeight, imgWidth) = (getImageSize(img)(1), getImageSize(img)(0))
////    val top = (imgHeight - outputSize(0)) / 2
////    val left = (imgWidth - outputSize(1)) / 2
////    crop(img, top, left, outputSize(0), outputSize(1))
////  }
//
//  def resizedCrop(
//                   img: Tensor,
//                   top: Int,
//                   left: Int,
//                   height: Int,
//                   width: Int,
//                   size: Array[Int],
//                   interpolation: InterpolationMode = BILINEAR,
//                   antialias: Option[Boolean] = Some(true)
//                 ): Tensor = {
//    val croppedImg = crop(img, top, left, height, width)
//    resize(croppedImg, size, interpolation, antialias = antialias)
//  }
//  // 辅助函数，生成 5 个裁剪图像
//  def fiveCrop(img: Tensor, size: Array[Int]): Array[Tensor] = {
//    val (width, height) = (getImageSize(img)(0), getImageSize(img)(1))
//    val cropHeight = size(0)
//    val cropWidth = size(1)
//
//    val topLeft = crop(img, 0, 0, cropHeight, cropWidth)
//    val topRight = crop(img, 0, width - cropWidth, cropHeight, cropWidth)
//    val bottomLeft = crop(img, height - cropHeight, 0, cropHeight, cropWidth)
//    val bottomRight = crop(img, height - cropHeight, width - cropWidth, cropHeight, cropWidth)
//    val center = centerCrop(img, size)
//
//    Array(topLeft, topRight, bottomLeft, bottomRight, center)
//  }
//
//  // ten_crop 函数
//  def tenCrop(img: Tensor, size: Array[Int], verticalFlip: Boolean = false): Array[Tensor] = {
//    val firstFive = fiveCrop(img, size)
//    val flippedImg = if (verticalFlip) vflip(img) else hflip(img)
//    val secondFive = fiveCrop(flippedImg, size)
//    firstFive ++ secondFive
//  }
//
//  // adjust_brightness 函数
//  def adjustBrightness(img: Tensor, brightnessFactor: Float): Tensor = {
//    if (brightnessFactor == 1.0f) {
//      img
//    } else {
//      val adjusted = img.mul(brightnessFactor)
//      torchGlobal.clamp(adjusted, 0.0f, 1.0f)
//    }
//  }
//
//  // adjust_contrast 函数
//  def adjustContrast(img: Tensor, contrastFactor: Float): Tensor = {
//    if (contrastFactor == 1.0f) {
//      img
//    } else {
//      val mean = torchGlobal.mean(img)
//      val adjusted = img.sub(mean).mul(contrastFactor).add(mean)
//      torchGlobal.clamp(adjusted, 0.0f, 1.0f)
//    }
//  }
//
//  // adjust_saturation 函数
//  def adjustSaturation(img: Tensor, saturationFactor: Float): Tensor = {
//    if (saturationFactor == 1.0f) {
//      img
//    } else {
//      val grayImg = rgbToGrayscale(img, 3)
//      val adjusted = grayImg.mul(1 - saturationFactor).add(img.mul(saturationFactor))
//      torchGlobal.clamp(adjusted, 0.0f, 1.0f)
//    }
//  }
//
//  // adjust_hue 函数
//  def adjustHue(img: Tensor, hueFactor: Float): Tensor = {
//    require(hueFactor >= -0.5 && hueFactor <= 0.5, "hueFactor must be in [-0.5, 0.5]")
//    if (hueFactor == 0.0f) {
//      img
//    } else {
//      val hsv = torchGlobal.rgb_to_hsv(img)
//      val hChannel = hsv.index({0}, 0)
//      val newH = hChannel.add(hueFactor).fmod(1.0f)
//      hsv.index_put_({0}, 0, newH)
//      torchGlobal.hsv_to_rgb(hsv)
//    }
//  }
//
//  // adjust_gamma 函数
//  def adjustGamma(img: Tensor, gamma: Float, gain: Float = 1.0f): Tensor = {
//    require(gamma >= 0, "Gamma must be non - negative")
//    if (gamma == 1.0f) {
//      img
//    } else {
//      val adjusted = img.pow(gamma).mul(gain)
//      torchGlobal.clamp(adjusted, 0.0f, 1.0f)
//    }
//  }
//
//  // _get_inverse_affine_matrix 函数
//  def _getInverseAffineMatrix(
//    center: Array[Float],
//    angle: Float,
//    translate: Array[Float],
//    scale: Float,
//    shear: Array[Float],
//    inverted: Boolean = true
//  ): Array[Float] = {
//    val angleRad = math.toRadians(angle).toFloat
//    val shearXRad = math.toRadians(shear(0)).toFloat
//    val shearYRad = if (shear.length > 1) math.toRadians(shear(1)).toFloat else 0.0f
//
//    val cosA = math.cos(angleRad).toFloat
//    val sinA = math.sin(angleRad).toFloat
//    val cosSX = math.cos(shearXRad).toFloat
//    val sinSX = math.sin(shearXRad).toFloat
//    val cosSY = math.cos(shearYRad).toFloat
//    val sinSY = math.sin(shearYRad).toFloat
//
//    val a = scale * (cosA * cosSY - sinA * sinSX * sinSY) / cosSY
//    val b = -scale * (cosA * sinSY + sinA * sinSX * cosSY) / cosSY
//    val c = center(0) - (a * center(0) + b * center(1)) + translate(0)
//    val d = scale * (sinA * cosSY + cosA * sinSX * sinSY) / cosSY
//    val e = scale * (sinA * sinSY - cosA * sinSX * cosSY) / cosSY
//    val f = center(1) - (d * center(0) + e * center(1)) + translate(1)
//
//    Array(a, b, c, d, e, f)
//  }
//
//  // rotate 函数
//  def rotate(
//    img: Tensor,
//    angle: Float,
//    interpolation: InterpolationMode = NEAREST,
//    expand: Boolean = false,
//    center: Array[Int] = null,
//    fill: Array[Float] = null
//  ): Tensor = {
//    val imgSize = getImageSize(img)
//    val imgCenter = if (center == null) {
//      Array(imgSize(0) / 2.0f, imgSize(1) / 2.0f)
//    } else {
//      center.map(_.toFloat)
//    }
//
//    val matrix = _getInverseAffineMatrix(imgCenter, angle, Array(0, 0), 1.0f, Array(0, 0))
//    val grid = torchGlobal.affine_grid(
//      torchGlobal.tensor(Array(matrix)).view(1, 2, 3),
//      img.unsqueeze(0).shape(),
//      align_corners = false
//    )
//
//    val mode = interpolation match {
//      case NEAREST => "nearest"
//      case BILINEAR => "bilinear"
//      case _ => throw new IllegalArgumentException("Unsupported interpolation mode")
//    }
//
//    val output = torchGlobal.grid_sample(
//      img.unsqueeze(0),
//      grid,
//      mode = mode,
//      padding_mode = "zeros",
//      align_corners = false
//    ).squeeze(0)
//
//    output
//  }
//
//  // affine 函数
//  def affine(
//    img: Tensor,
//    angle: Float,
//    translate: Array[Int],
//    scale: Float,
//    shear: Array[Float],
//    interpolation: InterpolationMode = NEAREST,
//    fill: Array[Float] = null,
//    center: Array[Int] = null
//  ): Tensor = {
//    val imgSize = getImageSize(img)
//    val imgCenter = if (center == null) {
//      Array(imgSize(0) / 2.0f, imgSize(1) / 2.0f)
//    } else {
//      center.map(_.toFloat)
//    }
//
//    val matrix = _getInverseAffineMatrix(
//      imgCenter,
//      angle,
//      translate.map(_.toFloat),
//      scale,
//      shear
//    )
//
//    val grid = torchGlobal.affine_grid(
//      torchGlobal.tensor(Array(matrix)).view(1, 2, 3),
//      img.unsqueeze(0).shape(),
//      align_corners = false
//    )
//
//    val mode = interpolation match {
//      case NEAREST => "nearest"
//      case BILINEAR => "bilinear"
//      case _ => throw new IllegalArgumentException("Unsupported interpolation mode")
//    }
//
//    val output = torchGlobal.grid_sample(
//      img.unsqueeze(0),
//      grid,
//      mode = mode,
//      padding_mode = "zeros",
//      align_corners = false
//    ).squeeze(0)
//
//    output
//  }
//
//  // to_grayscale 函数（假设不支持 Tensor，仅作占位）
//  def toGrayscale(img: Tensor, numOutputChannels: Int = 1): Tensor = {
//    throw new UnsupportedOperationException("toGrayscale does not support torch Tensor in this implementation")
//  }
//
//  // rgb_to_grayscale 函数
//  def rgbToGrayscale(img: Tensor, numOutputChannels: Int = 1): Tensor = {
//    val weights = torchGlobal.tensor(Array(0.299f, 0.587f, 0.114f)).view(3, 1, 1)
//    val grayImg = torchGlobal.sum(img * weights, dim = -3, keepdim = true)
//    if (numOutputChannels == 3) {
//      grayImg.expand(-1, 3, -1, -1)
//    } else {
//      grayImg
//    }
//  }
//
//  // erase 函数
//  def erase(img: Tensor, i: Int, j: Int, h: Int, w: Int, v: Tensor, inplace: Boolean = false): Tensor = {
//    val result = if (inplace) img else img.clone()
//    result.slice(-2, i, i + h).slice(-1, j, j + w).fill_(v)
//    result
//  }
//
//  // gaussian_blur 函数
//  def gaussianBlur(img: Tensor, kernelSize: Array[Int], sigma: Array[Float] = null): Tensor = {
//    val sigmaValue = if (sigma == null) {
//      Array(0.3f * ((kernelSize(0) - 1) * 0.5f - 1) + 0.8f, 0.3f * ((kernelSize(1) - 1) * 0.5f - 1) + 0.8f)
//    } else {
//      sigma
//    }
//
//    val output = torchGlobal.nn.functional.gaussian_blur(
//      img.unsqueeze(0),
//      kernelSize,
//      sigmaValue
//    ).squeeze(0)
//
//    output
//  }
//
//  // invert 函数
//  def invert(img: Tensor): Tensor = {
//    1.0f - img
//  }
//
//  // posterize 函数
//  def posterize(img: Tensor, bits: Int): Tensor = {
//    val shift = 8 - bits
//    val scaled = img.mul(255).toType(torchGlobal.UInt8)
//    val shifted = scaled.bitwise_right_shift(shift).bitwise_left_shift(shift)
//    shifted.toType(torchGlobal.Float).div(255.0f)
//  }
//
//  // solarize 函数
//  def solarize(img: Tensor, threshold: Float): Tensor = {
//    val mask = img.ge(threshold)
//    torchGlobal.where(mask, 1.0f - img, img)
//  }
//
//  // adjust_sharpness 函数
//  def adjustSharpness(img: Tensor, sharpnessFactor: Float): Tensor = {
//    if (sharpnessFactor == 1.0f) {
//      img
//    } else {
//      val blurred = gaussianBlur(img, Array(3, 3))
//      val adjusted = blurred.mul(1 - sharpnessFactor).add(img.mul(sharpnessFactor))
//      torchGlobal.clamp(adjusted, 0.0f, 1.0f)
//    }
//  }
//
//  // autocontrast 函数
//  def autocontrast(img: Tensor): Tensor = {
//    val minVal = torchGlobal.min(img)
//    val maxVal = torchGlobal.max(img)
//    if (minVal == maxVal) {
//      torchGlobal.fullLike(img, minVal)
//    } else {
//      val scale = 1.0f / (maxVal - minVal)
//      (img - minVal) * scale
//    }
//  }
//
//  // equalize 函数
//  def equalize(img: Tensor): Tensor = {
//    if (img.dtype() != torchGlobal.UInt8) {
//      throw new IllegalArgumentException("Input tensor must be of type torch.uint8")
//    }
//    val histogram = torchGlobal.histc(img, bins = 256, min = 0, max = 255)
//    val cdf = histogram.cumsum(0)
//    val cdfMin = cdf.min()
//    val cdfAdjusted = (cdf - cdfMin) * 255 / (cdf.max() - cdfMin)
//    val equalized = cdfAdjusted.index_select(0, img.view(-1).toType(torchGlobal.Long)).view(img.shape())
//    equalized
//  }
//
//  // elastic_transform 函数
//  def elasticTransform(
//    img: Tensor,
//    displacement: Tensor,
//    interpolation: InterpolationMode = BILINEAR,
//    fill: Array[Float] = null
//  ): Tensor = {
//    val imgShape = img.shape()
//    val requiredDispShape = Array(1, imgShape(imgShape.length - 2), imgShape(imgShape.length - 1), 2)
//    if (!displacement.shape().sameElements(requiredDispShape)) {
//      throw new IllegalArgumentException(s"Expected displacement shape ${requiredDispShape.mkString(", ")}, but got ${displacement.shape().mkString(", ")}")
//    }
//
//    val mode = interpolation match {
//      case NEAREST => "nearest"
//      case BILINEAR => "bilinear"
//      case _ => throw new IllegalArgumentException("Unsupported interpolation mode")
//    }
//
//    val grid = displacement.add(torchGlobal.meshgrid(
//      torchGlobal.linspace(-1.0f, 1.0f, imgShape(imgShape.length - 2)),
//      torchGlobal.linspace(-1.0f, 1.0f, imgShape(imgShape.length - 1))
//    ).reverse().map(_.unsqueeze(-1)).toArray)
//
//    val output = torchGlobal.grid_sample(
//      img.unsqueeze(0),
//      grid,
//      mode = mode,
//      padding_mode = "zeros",
//      align_corners = false
//    ).squeeze(0)
//
//    output
//  }
//}
