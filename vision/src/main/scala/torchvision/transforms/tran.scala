//package torchvision.transforms
//
//import org.bytedeco.pytorch.global.torch as torchGlobal
//import org.bytedeco.pytorch.*
//import java.util.EnumSet
//import scala.collection.mutable.ListBuffer
//import scala.util.Random
//
//// 定义插值模式枚举
//object InterpolationMode extends Enumeration {
//  type InterpolationMode = Value
//  val NEAREST, NEAREST_EXACT, BILINEAR, BICUBIC, BOX, HAMMING, LANCZOS = Value
//}
//
//import InterpolationMode._
//
//// 辅助函数类
//object Functional {
//  // 实现前面提到的 functional 中的方法
//  // ...
//}
//
//// Compose 类
//class Compose(transforms: List[Any]) {
//  def apply(img: Tensor): Tensor = {
//    var result = img
//    for (transform <- transforms) {
//      transform match {
//        case t: AnyRef if t.isInstanceOf[((Tensor) => Tensor)] =>
//          result = t.asInstanceOf[(Tensor) => Tensor](result)
//        case _ => throw new IllegalArgumentException("Unsupported transform type")
//      }
//    }
//    result
//  }
//}
//
//// ToTensor 类
//class ToTensor {
//  def apply(pic: Tensor): Tensor = {
//    // 假设 pic 已经是合适的格式，可能需要更多处理
//    pic
//  }
//}
//
//// PILToTensor 类
//class PILToTensor {
//  def apply(pic: Tensor): Tensor = {
//    pic.clone()
//  }
//}
//
//// ConvertImageDtype 类
//class ConvertImageDtype(dtype: ScalarType) extends torchGlobal.nn.Module {
//  override def forward(image: Tensor): Tensor = {
//    image.to(dtype)
//  }
//}
//
//// ToPILImage 类（此处仅占位，Scala 中无直接对应 PIL 库）
//class ToPILImage(mode: Option[String] = None) {
//  def apply(pic: Tensor): Any = {
//    throw new UnsupportedOperationException("ToPILImage is not fully implemented in Scala")
//  }
//}
//
//// Normalize 类
//class Normalize(mean: Array[Float], std: Array[Float], inplace: Boolean = false) extends torchGlobal.nn.Module {
//  override def forward(tensor: Tensor): Tensor = {
//    val result = if (inplace) tensor else tensor.clone()
//    val channels = result.sizes()(0)
//    for (i <- 0 until channels) {
//      result.index({0}, i).sub_(mean(i)).div_(std(i))
//    }
//    result
//  }
//}
//
//// Resize 类
//class Resize(size: Array[Int], interpolation: InterpolationMode = BILINEAR, maxSize: Option[Int] = None, antialias: Boolean = true) extends torchGlobal.nn.Module {
//  override def forward(img: Tensor): Tensor = {
//    val newSize = Functional._computeResizedOutputSize((img.sizes()(img.sizes().length - 2), img.sizes()(img.sizes().length - 1)), Some(size), maxSize)
//    val mode = interpolation match {
//      case NEAREST => "nearest"
//      case BILINEAR => "bilinear"
//      case _ => throw new IllegalArgumentException("Unsupported interpolation mode")
//    }
//    torchGlobal.nn.functional.interpolate(img.unsqueeze(0), newSize, mode = mode).squeeze(0)
//  }
//}
//
//// CenterCrop 类
//class CenterCrop(size: Array[Int]) extends torchGlobal.nn.Module {
//  override def forward(img: Tensor): Tensor = {
//    val imgHeight = img.sizes()(img.sizes().length - 2)
//    val imgWidth = img.sizes()(img.sizes().length - 1)
//    val top = (imgHeight - size(0)) / 2
//    val left = (imgWidth - size(1)) / 2
//    img.slice(-2, top, top + size(0)).slice(-1, left, left + size(1))
//  }
//}
//
//// Pad 类
//class Pad(padding: Array[Int], fill: Float = 0, paddingMode: String = "constant") extends torchGlobal.nn.Module {
//  override def forward(img: Tensor): Tensor = {
//    val padSizes = new Array[Int](padding.length * 2)
//    for (i <- padding.indices) {
//      padSizes(2 * i) = padding(i)
//      padSizes(2 * i + 1) = padding(i)
//    }
//    torchGlobal.nn.functional.pad(img, padSizes, paddingMode, fill)
//  }
//}
//
//// Lambda 类
//class Lambda(lambd: (Tensor) => Tensor) {
//  def apply(img: Tensor): Tensor = {
//    lambd(img)
//  }
//}
//
//// RandomApply 类
//class RandomApply(transforms: List[Any], p: Float = 0.5f) extends torchGlobal.nn.Module {
//  override def forward(img: Tensor): Tensor = {
//    if (Random.nextFloat() < p) {
//      var result = img
//      for (transform <- transforms) {
//        transform match {
//          case t: AnyRef if t.isInstanceOf[((Tensor) => Tensor)] =>
//            result = t.asInstanceOf[(Tensor) => Tensor](result)
//          case _ => throw new IllegalArgumentException("Unsupported transform type")
//        }
//      }
//      result
//    } else {
//      img
//    }
//  }
//}
//
//// RandomChoice 类
//class RandomChoice(transforms: List[Any], p: Option[Array[Float]] = None) {
//  def apply(img: Tensor): Tensor = {
//    val index = p match {
//      case Some(probs) =>
//        val r = Random.nextFloat()
//        var sum = 0.0f
//        for (i <- probs.indices) {
//          sum += probs(i)
//          if (r < sum) return i
//        }
//        probs.length - 1
//      case None => Random.nextInt(transforms.length)
//    }
//    val transform = transforms(index)
//    transform match {
//      case t: AnyRef if t.isInstanceOf[((Tensor) => Tensor)] =>
//        t.asInstanceOf[(Tensor) => Tensor](img)
//      case _ => throw new IllegalArgumentException("Unsupported transform type")
//    }
//  }
//}
//
//// RandomOrder 类
//class RandomOrder(transforms: List[Any]) {
//  def apply(img: Tensor): Tensor = {
//    val shuffledTransforms = Random.shuffle(transforms)
//    var result = img
//    for (transform <- shuffledTransforms) {
//      transform match {
//        case t: AnyRef if t.isInstanceOf[((Tensor) => Tensor)] =>
//          result = t.asInstanceOf[(Tensor) => Tensor](result)
//        case _ => throw new IllegalArgumentException("Unsupported transform type")
//      }
//    }
//    result
//  }
//}
//
//// RandomCrop 类
//class RandomCrop(size: Array[Int], padding: Option[Array[Int]] = None, padIfNeeded: Boolean = false, fill: Float = 0, paddingMode: String = "constant") extends torchGlobal.nn.Module {
//  def getParams(img: Tensor, outputSize: Array[Int]): (Int, Int, Int, Int) = {
//    val imgHeight = img.sizes()(img.sizes().length - 2)
//    val imgWidth = img.sizes()(img.sizes().length - 1)
//    val h = outputSize(0)
//    val w = outputSize(1)
//    val i = Random.nextInt(imgHeight - h + 1)
//    val j = Random.nextInt(imgWidth - w + 1)
//    (i, j, h, w)
//  }
//
//  override def forward(img: Tensor): Tensor = {
//    val (i, j, h, w) = getParams(img, size)
//    img.slice(-2, i, i + h).slice(-1, j, j + w)
//  }
//}
//
//// RandomHorizontalFlip 类
//class RandomHorizontalFlip(p: Float = 0.5f) extends torchGlobal.nn.Module {
//  override def forward(img: Tensor): Tensor = {
//    if (Random.nextFloat() < p) {
//      torchGlobal.flip(img, -1)
//    } else {
//      img
//    }
//  }
//}
//
//// RandomVerticalFlip 类
//class RandomVerticalFlip(p: Float = 0.5f) extends torchGlobal.nn.Module {
//  override def forward(img: Tensor): Tensor = {
//    if (Random.nextFloat() < p) {
//      torchGlobal.flip(img, -2)
//    } else {
//      img
//    }
//  }
//}
//
//// RandomResizedCrop 类
//class RandomResizedCrop(size: Array[Int], scale: (Float, Float) = (0.08f, 1.0f), ratio: (Float, Float) = (3.0f / 4.0f, 4.0f / 3.0f), interpolation: InterpolationMode = BILINEAR, antialias: Option[Boolean] = Some(true)) extends torchGlobal.nn.Module {
//  def getParams(img: Tensor, scale: (Float, Float), ratio: (Float, Float)): (Int, Int, Int, Int) = {
//    val imgHeight = img.sizes()(img.sizes().length - 2)
//    val imgWidth = img.sizes()(img.sizes().length - 1)
//    val area = imgHeight * imgWidth
//
//    for (_ <- 0 until 10) {
//      val targetArea = area * Random.nextFloat() * (scale._2 - scale._1) + scale._1 * area
//      val aspectRatio = Math.exp(Math.log(ratio._1) + Random.nextFloat() * (Math.log(ratio._2) - Math.log(ratio._1))).toFloat
//
//      val w = Math.round(Math.sqrt(targetArea * aspectRatio)).toInt
//      val h = Math.round(Math.sqrt(targetArea / aspectRatio)).toInt
//
//      if (w <= imgWidth && h <= imgHeight) {
//        val i = Random.nextInt(imgHeight - h + 1)
//        val j = Random.nextInt(imgWidth - w + 1)
//        return (i, j, h, w)
//      }
//    }
//
//    val inRatio = imgWidth.toFloat / imgHeight.toFloat
//    if (inRatio < ratio._1) {
//      val w = imgWidth
//      val h = Math.round(w / ratio._1).toInt
//      val i = Math.max(0, (imgHeight - h) / 2)
//      val j = 0
//      (i, j, h, w)
//    } else if (inRatio > ratio._2) {
//      val h = imgHeight
//      val w = Math.round(h * ratio._2).toInt
//      val i = 0
//      val j = Math.max(0, (imgWidth - w) / 2)
//      (i, j, h, w)
//    } else {
//      val w = imgWidth
//      val h = imgHeight
//      val i = 0
//      val j = 0
//      (i, j, h, w)
//    }
//  }
//
//  override def forward(img: Tensor): Tensor = {
//    val (i, j, h, w) = getParams(img, scale, ratio)
//    val croppedImg = img.slice(-2, i, i + h).slice(-1, j, j + w)
//    val resizedImg = new Resize(size, interpolation).forward(croppedImg)
//    resizedImg
//  }
//}
//
//// FiveCrop 类
//class FiveCrop(size: Array[Int]) extends torchGlobal.nn.Module {
//  override def forward(img: Tensor): Array[Tensor] = {
//    val imgHeight = img.sizes()(img.sizes().length - 2)
//    val imgWidth = img.sizes()(img.sizes().length - 1)
//    val h = size(0)
//    val w = size(1)
//
//    val topLeft = img.slice(-2, 0, h).slice(-1, 0, w)
//    val topRight = img.slice(-2, 0, h).slice(-1, imgWidth - w, imgWidth)
//    val bottomLeft = img.slice(-2, imgHeight - h, imgHeight).slice(-1, 0, w)
//    val bottomRight = img.slice(-2, imgHeight - h, imgHeight).slice(-1, imgWidth - w, imgWidth)
//    val center = new CenterCrop(size).forward(img)
//
//    Array(topLeft, topRight, bottomLeft, bottomRight, center)
//  }
//}
//
//// TenCrop 类
//class TenCrop(size: Array[Int], verticalFlip: Boolean = false) extends torchGlobal.nn.Module {
//  override def forward(img: Tensor): Array[Tensor] = {
//    val fiveCrops = new FiveCrop(size).forward(img)
//    val flippedImg = if (verticalFlip) torchGlobal.flip(img, -2) else torchGlobal.flip(img, -1)
//    val flippedFiveCrops = new FiveCrop(size).forward(flippedImg)
//    fiveCrops ++ flippedFiveCrops
//  }
//}
//
//// LinearTransformation 类
//class LinearTransformation(transformationMatrix: Tensor, meanVector: Tensor) extends torchGlobal.nn.Module {
//  override def forward(tensor: Tensor): Tensor = {
//    val originalShape = tensor.shape()
//    val flattened = tensor.flatten()
//    val centered = flattened.sub(meanVector)
//    val transformed = torchGlobal.mm(centered.unsqueeze(0), transformationMatrix).squeeze(0)
//    transformed.view(originalShape)
//  }
//}
//
//// ColorJitter 类
//class ColorJitter(
//  brightness: Either[Float, (Float, Float)] = Left(0),
//  contrast: Either[Float, (Float, Float)] = Left(0),
//  saturation: Either[Float, (Float, Float)] = Left(0),
//  hue: Either[Float, (Float, Float)] = Left(0)
//) extends torchGlobal.nn.Module {
//  def getParams(
//    brightness: Option[(Float, Float)],
//    contrast: Option[(Float, Float)],
//    saturation: Option[(Float, Float)],
//    hue: Option[(Float, Float)]
//  ): (List[((Tensor) => Tensor)], Float, Float, Float, Float) = {
//    val brightnessFactor = brightness.map { case (min, max) => min + Random.nextFloat() * (max - min) }.getOrElse(1.0f)
//    val contrastFactor = contrast.map { case (min, max) => min + Random.nextFloat() * (max - min) }.getOrElse(1.0f)
//    val saturationFactor = saturation.map { case (min, max) => min + Random.nextFloat() * (max - min) }.getOrElse(1.0f)
//    val hueFactor = hue.map { case (min, max) => min + Random.nextFloat() * (max - min) }.getOrElse(0.0f)
//
//    val transforms = ListBuffer[((Tensor) => Tensor)]()
//
//    if (brightness.isDefined) {
//      transforms += (img => Functional.adjustBrightness(img, brightnessFactor))
//    }
//    if (contrast.isDefined) {
//      transforms += (img => Functional.adjustContrast(img, contrastFactor))
//    }
//    if (saturation.isDefined) {
//      transforms += (img => Functional.adjustSaturation(img, saturationFactor))
//    }
//    if (hue.isDefined) {
//      transforms += (img => Functional.adjustHue(img, hueFactor))
//    }
//
//    (Random.shuffle(transforms.toList), brightnessFactor, contrastFactor, saturationFactor, hueFactor)
//  }
//
//  override def forward(img: Tensor): Tensor = {
//    val (transforms, _, _, _, _) = getParams(
//      brightness.right.toOption,
//      contrast.right.toOption,
//      saturation.right.toOption,
//      hue.right.toOption
//    )
//    var result = img
//    for (transform <- transforms) {
//      result = transform(result)
//    }
//    result
//  }
//}
//
//// RandomRotation 类
//class RandomRotation(degrees: Either[Float, (Float, Float)], interpolation: InterpolationMode = NEAREST, expand: Boolean = false, center: Option[Array[Int]] = None, fill: Float = 0) extends torchGlobal.nn.Module {
//  def getParams(degrees: (Float, Float)): Float = {
//    degrees._1 + Random.nextFloat() * (degrees._2 - degrees._1)
//  }
//
//  override def forward(img: Tensor): Tensor = {
//    val angle = degrees match {
//      case Left(d) => Random.nextFloat() * 2 * d - d
//      case Right(d) => getParams(d)
//    }
//    Functional.rotate(img, angle, interpolation, expand, center.map(_.map(_.toFloat)), Array(fill))
//  }
//}
//
//// RandomAffine 类
//class RandomAffine(
//  degrees: Either[Float, (Float, Float)],
//  translate: Option[(Float, Float)] = None,
//  scale: Option[(Float, Float)] = None,
//  shear: Option[Either[Float, (Float, Float)]] = None,
//  interpolation: InterpolationMode = NEAREST,
//  fill: Float = 0,
//  center: Option[Array[Int]] = None
//) extends torchGlobal.nn.Module {
//  override def forward(img: Tensor): Tensor = {
//    val angle = degrees match {
//      case Left(d) => Random.nextFloat() * 2 * d - d
//      case Right(d) => d._1 + Random.nextFloat() * (d._2 - d._1)
//    }
//
//    val translateValues = translate.map { case (tx, ty) =>
//      val imgWidth = img.sizes()(img.sizes().length - 1)
//      val imgHeight = img.sizes()(img.sizes().length - 2)
//      (Random.nextFloat() * 2 * tx * imgWidth - tx * imgWidth, Random.nextFloat() * 2 * ty * imgHeight - ty * imgHeight)
//    }.getOrElse((0.0f, 0.0f))
//
//    val scaleValue = scale.map { case (min, max) => min + Random.nextFloat() * (max - min) }.getOrElse(1.0f)
//
//    val shearValue = shear.map {
//      case Left(s) => Random.nextFloat() * 2 * s - s
//      case Right(s) => s._1 + Random.nextFloat() * (s._2 - s._1)
//    }.getOrElse(0.0f)
//
//    Functional.affine(
//      img,
//      angle,
//      Array(translateValues._1.toInt, translateValues._2.toInt),
//      scaleValue,
//      Array(shearValue),
//      interpolation,
//      Array(fill),
//      center.map(_.map(_.toFloat))
//    )
//  }
//}
//
//// Grayscale 类
//class Grayscale(numOutputChannels: Int = 1) extends torchGlobal.nn.Module {
//  override def forward(img: Tensor): Tensor = {
//    Functional.rgbToGrayscale(img, numOutputChannels)
//  }
//}
//
//// RandomGrayscale 类
//class RandomGrayscale(p: Float = 0.1f) extends torchGlobal.nn.Module {
//  override def forward(img: Tensor): Tensor = {
//    if (Random.nextFloat() < p) {
//      new Grayscale(3).forward(img)
//    } else {
//      img
//    }
//  }
//}
//
//// RandomPerspective 类
//class RandomPerspective(distortionScale: Float = 0.5f, p: Float = 0.5f, interpolation: InterpolationMode = BILINEAR, fill: Float = 0) extends torchGlobal.nn.Module {
//  override def forward(img: Tensor): Tensor = {
//    if (Random.nextFloat() < p) {
//      val (startPoints, endPoints) = Functional.getRandomPerspectiveParams(img.sizes()(img.sizes().length - 1), img.sizes()(img.sizes().length - 2), distortionScale)
//      Functional.perspective(img, startPoints, endPoints, interpolation, fill)
//    } else {
//      img
//    }
//  }
//}
//
//// RandomErasing 类
//class RandomErasing(p: Float = 0.5f, scale: (Float, Float) = (0.02f, 0.33f), ratio: (Float, Float) = (0.3f, 3.3f), value: Float = 0) extends torchGlobal.nn.Module {
//  override def forward(img: Tensor): Tensor = {
//    if (Random.nextFloat() < p) {
//      val imgHeight = img.sizes()(img.sizes().length - 2)
//      val imgWidth = img.sizes()(img.sizes().length - 1)
//      val area = imgHeight * imgWidth
//
//      for (_ <- 0 until 10) {
//        val targetArea = area * Random.nextFloat() * (scale._2 - scale._1) + scale._1 * area
//        val aspectRatio = Math.exp(Math.log(ratio._1) + Random.nextFloat() * (Math.log(ratio._2) - Math.log(ratio._1))).toFloat
//
//        val h = Math.round(Math.sqrt(targetArea / aspectRatio)).toInt
//        val w = Math.round(Math.sqrt(targetArea * aspectRatio)).toInt
//
//        if (h <= imgHeight && w <= imgWidth) {
//          val i = Random.nextInt(imgHeight - h + 1)
//          val j = Random.nextInt(imgWidth - w + 1)
//          return Functional.erase(img, i, j, h, w, torchGlobal.full(img.shape(), value), true)
//        }
//      }
//    }
//    img
//  }
//}
//
//// GaussianBlur 类
//class GaussianBlur(kernelSize: Array[Int], sigma: Option[Array[Float]] = None) extends torchGlobal.nn.Module {
//  override def forward(img: Tensor): Tensor = {
//    Functional.gaussianBlur(img, kernelSize, sigma.orNull)
//  }
//}
//
//// RandomInvert 类
//class RandomInvert(p: Float = 0.5f) extends torchGlobal.nn.Module {
//  override def forward(img: Tensor): Tensor = {
//    if (Random.nextFloat() < p) {
//      Functional.invert(img)
//    } else {
//      img
//    }
//  }
//}
//
//// RandomPosterize 类
//class RandomPosterize(bits: Int, p: Float = 0.5f) extends torchGlobal.nn.Module {
//  override def forward(img: Tensor): Tensor = {
//    if (Random.nextFloat() < p) {
//      Functional.posterize(img, bits)
//    } else {
//      img
//    }
//  }
//}
//
//// RandomSolarize 类
//class RandomSolarize(threshold: Float, p: Float = 0.5f) extends torchGlobal.nn.Module {
//  override def forward(img: Tensor): Tensor = {
//    if (Random.nextFloat() < p) {
//      Functional.solarize(img, threshold)
//    } else {
//      img
//    }
//  }
//}
//
//// RandomAdjustSharpness 类
//class RandomAdjustSharpness(sharpnessFactor: Float, p: Float = 0.5f) extends torchGlobal.nn.Module {
//  override def forward(img: Tensor): Tensor = {
//    if (Random.nextFloat() < p) {
//      Functional.adjustSharpness(img, sharpnessFactor)
//    } else {
//      img
//    }
//  }
//}
//
//// RandomAutocontrast 类
//class RandomAutocontrast(p: Float = 0.5f) extends torchGlobal.nn.Module {
//  override def forward(img: Tensor): Tensor = {
//    if (Random.nextFloat() < p) {
//      Functional.autocontrast(img)
//    } else {
//      img
//    }
//  }
//}
//
//// RandomEqualize 类
//class RandomEqualize(p: Float = 0.5f) extends torchGlobal.nn.Module {
//  override def forward(img: Tensor): Tensor = {
//    if (Random.nextFloat() < p) {
//      Functional.equalize(img)
//    } else {
//      img
//    }
//  }
//}
//
//class RandomPerspective(
//                         distortionScale: Float = 0.5f,
//                         p: Float = 0.5f,
//                         interpolation: InterpolationMode = BILINEAR,
//                         fill: Float = 0
//                       ) extends torchGlobal.nn.Module {
//
//  def getParams(width: Int, height: Int, distortionScale: Float): (Array[Array[Float]], Array[Array[Float]]) = {
//    val halfHeight = height / 2.0f
//    val halfWidth = width / 2.0f
//
//    val startPoints = Array(
//      Array(0.0f, 0.0f),
//      Array(width.toFloat, 0.0f),
//      Array(width.toFloat, height.toFloat),
//      Array(0.0f, height.toFloat)
//    )
//
//    val randomDistortion = distortionScale * Random.nextFloat()
//    val endPoints = Array(
//      Array(
//        startPoints(0)(0) + Random.nextFloat() * width * randomDistortion,
//        startPoints(0)(1) + Random.nextFloat() * height * randomDistortion
//      ),
//      Array(
//        startPoints(1)(0) - Random.nextFloat() * width * randomDistortion,
//        startPoints(1)(1) + Random.nextFloat() * height * randomDistortion
//      ),
//      Array(
//        startPoints(2)(0) - Random.nextFloat() * width * randomDistortion,
//        startPoints(2)(1) - Random.nextFloat() * height * randomDistortion
//      ),
//      Array(
//        startPoints(3)(0) + Random.nextFloat() * width * randomDistortion,
//        startPoints(3)(1) - Random.nextFloat() * height * randomDistortion
//      )
//    )
//
//    (startPoints, endPoints)
//  }
//
//  def getPerspectiveTransform(startPoints: Array[Array[Float]], endPoints: Array[Array[Float]]): Tensor = {
//    val src = torchGlobal.tensor(startPoints.flatten).view(4, 2)
//    val dst = torchGlobal.tensor(endPoints.flatten).view(4, 2)
//    torchGlobal.get_perspective_transform(src, dst)
//  }
//
//  override def forward(img: Tensor): Tensor = {
//    if (Random.nextFloat() < p) {
//      val (width, height) = (img.sizes()(img.sizes().length - 1), img.sizes()(img.sizes().length - 2))
//      val (startPoints, endPoints) = getParams(width, height, distortionScale)
//      val transformMatrix = getPerspectiveTransform(startPoints, endPoints)
//
//      val mode = interpolation match {
//        case NEAREST => "nearest"
//        case BILINEAR => "bilinear"
//      }
//
//      val grid = torchGlobal.affine_grid(
//        transformMatrix.unsqueeze(0),
//        img.unsqueeze(0).shape(),
//        align_corners = false
//      )
//
//      val output = torchGlobal.grid_sample(
//        img.unsqueeze(0),
//        grid,
//        mode = mode,
//        padding_mode = "zeros",
//        align_corners = false
//      ).squeeze(0)
//
//      output
//    } else {
//      img
//    }
//  }
//
//  override def toString: String = {
//    s"RandomPerspective(distortionScale=$distortionScale, p=$p, interpolation=$interpolation, fill=$fill)"
//  }
//}
//
//abstract class RandomTransforms(transforms: Seq[Any]) {
//  def apply(args: Any*): Any = throw new NotImplementedError()
//
//  override def toString: String = {
//    s"RandomTransforms(transforms=${transforms.mkString(", ")})"
//  }
//}
//
//// ElasticTransform 类
//class ElasticTransform(
//                        alpha: Float = 50.0f,
//                        sigma: Float = 5.0f,
//                        interpolation: InterpolationMode = BILINEAR,
//                        fill: Float = 0
//                      ) extends torchGlobal.nn.Module {
//
//  def getParams(alpha: Float, sigma: Float, height: Int, width: Int): Tensor = {
//    val dx = torchGlobal.rand(1, 1, height, width).mul(2).sub(1)
//    val kx = (8 * sigma + 1).toInt + (if ((8 * sigma + 1).toInt % 2 == 0) 1 else 0)
//    val blurredDx = if (sigma > 0.0f) {
//      torchGlobal.nn.functional.gaussianBlur(dx, Array(kx, kx), Array(sigma, sigma))
//    } else {
//      dx
//    }
//    val scaledDx = blurredDx.mul(alpha / height)
//
//    val dy = torchGlobal.rand(1, 1, height, width).mul(2).sub(1)
//    val ky = (8 * sigma + 1).toInt + (if ((8 * sigma + 1).toInt % 2 == 0) 1 else 0)
//    val blurredDy = if (sigma > 0.0f) {
//      torchGlobal.nn.functional.gaussianBlur(dy, Array(ky, ky), Array(sigma, sigma))
//    } else {
//      dy
//    }
//    val scaledDy = blurredDy.mul(alpha / width)
//
//    torchGlobal.cat(Array(scaledDx, scaledDy), 1).permute(0, 2, 3, 1)
//  }
//
//  override def forward(img: Tensor): Tensor = {
//    val height = img.size(-2)
//    val width = img.size(-1)
//    val displacement = getParams(alpha, sigma, height, width)
//
//    val mode = interpolation match {
//      case NEAREST => "nearest"
//      case BILINEAR => "bilinear"
//    }
//
//    val grid = displacement.add(torchGlobal.meshgrid(
//      torchGlobal.linspace(-1.0f, 1.0f, height),
//      torchGlobal.linspace(-1.0f, 1.0f, width)
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
//
//  override def toString: String = {
//    s"ElasticTransform(alpha=$alpha, sigma=$sigma, interpolation=$interpolation, fill=$fill)"
//  }
//}
