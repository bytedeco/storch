package torch.nn.modules

import org.bytedeco.javacpp.{DoublePointer, LongPointer}
import org.bytedeco.pytorch.{ConstantPad2dImpl, ConstantPad2dOptions, LPPool3dImpl, LPPool3dOptions}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn
import torch.nn.modules.attention.PositionalEncoding
import torch.nn.modules.attention.Transformer.TransformerActivation.kGELU
import torch.nn.modules.pooling.AdaptiveAvgPool2d
import torch.nn.modules.regularization.Upsample

class LPPool3dRawSuite extends munit.FunSuite {
  test("LPPool3dRawSuite output shapes") {
    // java.lang.RuntimeException: integer out of range
    val kernelSize = (3, 2, 2)
    val normType = 1.2f
    val stride = (2, 1, 2)
    val ceilMode = false
    System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
    System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
    val options: LPPool3dOptions = LPPool3dOptions(toNative(kernelSize))
    //  private val options: LPPool3dOptions = kernelSize match {
    //    case k: Int             => LPPool3dOptions(toNative((k, k, k)))
    //    case k: (Int, Int, Int) => LPPool3dOptions(toNative(k))
    //  }
    val input = torch.randn(Seq(20, 16, 50, 44, 31))
    stride match {
      //      case s: Int => options.stride().put(Array(s.toLong, s.toLong, s.toLong) *)
      case s: (Int, Int, Int) => options.stride().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
    }
    kernelSize match {
      //      case s: Int => options.kernel_size().put(Array(s.toLong, s.toLong, s.toLong) *)
      case s: (Int, Int, Int) =>
        options.kernel_size().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
    }

    options.ceil_mode().put(ceilMode)
    options.norm_type().put(DoublePointer(1).put(normType.toDouble))
    val nativeModule: LPPool3dImpl = LPPool3dImpl(options)
    val output = fromNative(nativeModule.forward(input.native))
    println(s" output shape ${output.shape}")

    //    val m12 = nn.LPPool3d(kernel_size = (3, 2, 2), norm_type = 1.2f, stride = (2, 1, 2))
    //    val input = torch.randn(Seq(20, 16, 50, 44, 31))
    //    println(m12(input).shape) //ArraySeq(20, 16, 24, 43, 15)
    //    //    ArraySeq(20, 16, 24, 43, 15)
    //    assertEquals(m12(input).shape, Seq(1, 64, 5, 7))
    //    val m22 = nn.LPPool3d(kernel_size = (1, 1), norm_type = 2, stride = Some(1))
    //    assertEquals(m22(input).shape, Seq(1, 64, 1, 1))
  }
}

//https://pytorch.org/docs/2.3/generated/torch.nn.ZeroPad1d.html  ConstantPad2dOptions
class ZeroPad1dSuite extends munit.FunSuite {
  test("ZeroPad1dSuite output shapes") {
    val m12 = nn.ZeroPad1d((2))
    val input = torch.randn(Seq(1, 2, 4))
    assertEquals(m12(input).shape, Seq(1, 2, 8))

    val input2 = torch.randn(Seq(1, 2, 3))
    assertEquals(m12(input2).shape, Seq(1, 2, 7))

    val m13 = nn.ZeroPad1d((3, 1))
    assertEquals(m13(input2).shape, Seq(1, 2, 7))
    //    val m22 = nn.AdaptiveMaxPool2d((1, 1))
    //    assertEquals(m22(input).shape, Seq(1, 64, 1, 1))
  }
}

//https://pytorch.org/docs/2.3/generated/torch.nn.ZeroPad2d.html
class ZeroPad2dSuite extends munit.FunSuite {
  test("ZeroPad2dSuite output shapes") {
    val m112 = nn.ZeroPad2d(2)
    val input1 = torch.randn(Seq(1, 1, 3, 3))
    println(m112(input1).shape) // Failed SymIntArrayRef expected to contain only concrete integers
    val m12 = nn.ZeroPad2d((1, 1, 2, 0))
    val input = torch.randn(Seq(1, 1, 3, 3))
    println(m12(input).shape) // ArraySeq(1, 1, 5, 5)  Success

  }
}

//https://pytorch.org/docs/2.3/generated/torch.nn.ZeroPad3d.html
class ZeroPad3dSuite extends munit.FunSuite {
  test("ZeroPad3dSuite output shapes") {
    val m12 = nn.ZeroPad3d(3)
    val input = torch.randn(Seq(16, 3, 10, 20, 20))
    println(m12(input).shape) // Failed SymIntArrayRef expected to contain only concrete integers
    val m123 = nn.ZeroPad3d((3, 3, 6, 6, 0, 1))
    val input3 = torch.randn(Seq(16, 3, 10, 20, 20))
    println(m123(input3).shape) // ArraySeq(16, 3, 11, 32, 26)  Success
    ////    assertEquals(m12(input).shape, Seq(1, 7, 7))
  }
}

//https://pytorch.org/docs/stable/generated/torch.nn.ReplicationPad3d.html
class ReplicationPad3d extends munit.FunSuite {
  test("ReplicationPad3d output shapes") {
    val m12 = nn.ReplicationPad3d(3)
    val input = torch.randn(Seq(16, 3, 8, 320, 480))
    println(m12(input).shape) // Failed SymIntArrayRef expected to contain only concrete integers

    ////    assertEquals(m12(input).shape, Seq(1, 7, 7))

    val m123 = nn.ReplicationPad3d((3, 3, 6, 6, 1, 1))
    val input3 = torch.randn(Seq(16, 3, 8, 320, 480))
    println(m123(input3).shape) // ArraySeq(16, 3, 10, 332, 486)  Success

  }
}

//https://pytorch.org/docs/stable/generated/torch.nn.ReplicationPad2d.html
class ReplicationPad2d extends munit.FunSuite {
  test("ReplicationPad2d output shapes") {
    val m12 = nn.ReplicationPad2d(2)
    val input1 = torch.arange(end = 9, dtype = torch.float32).reshape(1, 1, 3, 3)
    //    val input = torch.randn(Seq(16, 3, 8, 320, 480))
    println(m12(input1).shape) // Failed SymIntArrayRef expected to contain only concrete integers

    ////    assertEquals(m12(input).shape, Seq(1, 7, 7))
    val input2 = torch.arange(end = 9, dtype = torch.float32).reshape(1, 1, 3, 3)
    val m123 = nn.ReplicationPad2d((1, 1, 2, 0))
    //    val input3 = torch.randn(Seq(16, 3, 8, 320, 480))
    println(m123(input2).shape) // ArraySeq(1, 1, 5, 5)  Success

  }
}

//https://pytorch.org/docs/2.3/generated/torch.nn.ReplicationPad1d.html
class ReplicationPad1d extends munit.FunSuite {
  test("ReplicationPad1d output shapes") {
    val m12 = nn.ReplicationPad1d(2)
    val input1 = torch.arange(end = 8, dtype = torch.float32).reshape(1, 2, 4)
    //    val input = torch.randn(Seq(16, 3, 8, 320, 480))
    println(m12(input1).shape) // ArraySeq(1, 2, 8)  Success

    ////    assertEquals(m12(input).shape, Seq(1, 7, 7))
    val input2 = torch.arange(end = 8, dtype = torch.float32).reshape(1, 2, 4)
    val m123 = nn.ReplicationPad1d((3, 1))
    //    val input3 = torch.randn(Seq(16, 3, 8, 320, 480))
    println(m123(input2).shape) // ArraySeq(1, 2, 8)  Success

  }
}

//https://pytorch.org/docs/stable/generated/torch.nn.ReflectionPad3d.html
class ReflectionPad3d extends munit.FunSuite {
  test("ReflectionPad3d output shapes") {
    val m12 = nn.ReflectionPad3d(1)
    val input1 = torch.arange(end = 8, dtype = torch.float32).reshape(1, 1, 2, 2, 2)
    //    val input = torch.randn(Seq(16, 3, 8, 320, 480))
    println(
      m12(input1).shape
    ) // Failed SymIntArrayRef expected to contain only concrete integers  [1,1,4,4,4]  Success

    ////    assertEquals(m12(input).shape, Seq(1, 7, 7))
    //    val input2 = torch.arange(end = 8, dtype = torch.float32).reshape(1, 2, 4)
    val m123 = nn.ReplicationPad3d((3, 1, 4, 2, 2, 6))
    //    val input3 = torch.randn(Seq(16, 3, 8, 320, 480))
    println(m123(input1).shape) // ArraySeq(1, 2, 8)  Success

  }
}

//https://pytorch.org/docs/2.0/generated/torch.nn.ReflectionPad2d.html
class ReflectionPad2d extends munit.FunSuite {
  test("ReflectionPad2d output shapes") {
    val m12 = nn.ReflectionPad2d(2)
    val input1 = torch.arange(end = 9, dtype = torch.float32).reshape(1, 1, 3, 3)
    //    val input = torch.randn(Seq(16, 3, 8, 320, 480))
    println(
      m12(input1).shape
    ) // Failed SymIntArrayRef expected to contain only concrete integers  [1,1,7,7]  Success

    val m123 = nn.ReflectionPad2d((1, 1, 2, 0))
    val input13 = torch.arange(end = 9, dtype = torch.float32).reshape(1, 1, 3, 3)
    //    val input = torch.randn(Seq(16, 3, 8, 320, 480))
    println(m123(input13).shape) // ArraySeq(1, 1, 5, 5)  Success

  }
}

//https://pytorch.org/docs/2.3/generated/torch.nn.ReflectionPad1d.html
class ReflectionPad1d extends munit.FunSuite {
  test("ReflectionPad1d output shapes") {
    val m12 = nn.ReflectionPad1d(2)
    val input1 = torch.arange(end = 8, dtype = torch.float32).reshape(1, 2, 4)
    //    val input = torch.randn(Seq(16, 3, 8, 320, 480))
    println(m12(input1).shape) // ArraySeq(1, 2, 8)  Success

    val m123 = nn.ReflectionPad1d((3, 1))
    val input13 = torch.arange(end = 8, dtype = torch.float32).reshape(1, 2, 4)
    //    val input = torch.randn(Seq(16, 3, 8, 320, 480))
    println(m123(input13).shape) // ArraySeq(1, 2, 8)  Success

  }
}

//https://pytorch.org/docs/2.0/generated/torch.nn.ConstantPad3d.html
class ConstantPad3d extends munit.FunSuite {
  test("ConstantPad3d output shapes") {
    System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
    val m12 = nn.ConstantPad3d(3, 3.5)
    //    val input1 = torch.arange(end = 8, dtype = torch.float32).reshape(1, 2, 4)
    val input1 = torch.randn(Seq(16, 3, 10, 20, 30))
    println(m12(input1).shape) // Failed SymIntArrayRef expected to contain only concrete integers

    val m123 = nn.ConstantPad3d((3, 3, 6, 6, 0, 1), 3.5)
    val input13 = torch.randn(Seq(16, 3, 10, 20, 30))
    //    val input13 = torch.arange(end = 8, dtype = torch.float32).reshape(1, 2, 4)
    //    val input = torch.randn(Seq(16, 3, 8, 320, 480))
    println(m123(input13).shape) // ArraySeq(16, 3, 11, 32, 36)  Success

  }
}

class ConstantPad2dRaw extends munit.FunSuite {
  test("ConstantPad2d raw output shapes") {
    System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
    val padding = 2
    val value = 3.5d
    val paddingNative =
      LongPointer(Array(padding.toLong, padding.toLong, padding.toLong, padding.toLong)*)
    //    val paddingNative = padding match {
    //      case (top, bottom, left, right): (Int,Int,Int,Int) => toNative(top, bottom, left, right)
    //      case  (top :Int, bottom:Int) => toNative(top, top, bottom, bottom)
    //      case x: Int =>
    //        LongPointer(Array(x.toLong, x.toLong, x.toLong, x.toLong) *) // IntPointer(Array(x,x,x,x)*)
    //      case _ => throw new IllegalArgumentException("padding must be a tuple of 2, 4 or 8 integers")
    //    }
    val options: ConstantPad2dOptions = ConstantPad2dOptions(paddingNative)
    options.padding().put(paddingNative)
    value match {
      //      case x: Float => options.value().put(x.toDouble)
      case x: Double => options.value().put(x)
    }
    val input1 = torch.randn(Seq(1, 2, 2))
    val nativeModule: ConstantPad2dImpl = ConstantPad2dImpl(options)
    val output = fromNative(nativeModule.forward(input1.native))
    println(s"output shape ${output.shape}")
    //    val m12 = nn.ConstantPad2d(2, 3.5f)
    //
    //    //    val input1 = torch.arange(end = 8, dtype = torch.float32).reshape(1, 2, 4)
    //
    //    println(m12(input1).shape) // Failed SymIntArrayRef expected to contain only concrete integers [1,6,6]  Success
    //
    //
    //    val m123 = nn.ConstantPad2d((3, 0, 2, 1), 3.5)
    //    val input13 = torch.randn(Seq(1, 2, 2))
    //    //    val input13 = torch.arange(end = 8, dtype = torch.float32).reshape(1, 2, 4)
    //    //    val input = torch.randn(Seq(16, 3, 8, 320, 480))
    //    println(m123(input13).shape) // ArraySeq(1, 5, 5)  Success

  }
}

//java.lang.ClassCastException: class org.bytedeco.javacpp.DoublePointer cannot be cast to class scala.runtime.Nothing$ (org.bytedeco.javacpp.DoublePointer and scala.runtime.Nothing$ are in unnamed module of loader 'app')
//https://pytorch.org/docs/stable/generated/torch.nn.ConstantPad2d.html
class ConstantPad2d extends munit.FunSuite {
  test("ConstantPad2d output shapes") {
    System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
    val m12 = nn.ConstantPad2d(2, 3.5f)
    //    val input1 = torch.arange(end = 8, dtype = torch.float32).reshape(1, 2, 4)
    val input1 = torch.randn(Seq(1, 2, 2))
    println(
      m12(input1).shape
    ) // Failed SymIntArrayRef expected to contain only concrete integers [1,6,6]  Success

    val m123 = nn.ConstantPad2d((3, 0, 2, 1), 3.5)
    val input13 = torch.randn(Seq(1, 2, 2))
    //    val input13 = torch.arange(end = 8, dtype = torch.float32).reshape(1, 2, 4)
    //    val input = torch.randn(Seq(16, 3, 8, 320, 480))
    println(m123(input13).shape) // ArraySeq(1, 5, 5)  Success

  }
}

//https://pytorch.org/docs/stable/generated/torch.nn.ConstantPad1d.html
class ConstantPad1d extends munit.FunSuite {
  test("ConstantPad1d output shapes") {
    System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
    val m12 = nn.ConstantPad1d(2, 3.5f)
    //    val input1 = torch.arange(end = 8, dtype = torch.float32).reshape(1, 2, 4)
    val input1 = torch.randn(Seq(1, 2, 4))
    println(m12(input1).shape) // ArraySeq(1, 2, 8)  Success

    val m123 = nn.ConstantPad1d((3, 1), 3.5)
    val input13 = torch.randn(Seq(1, 2, 3))
    //    val input13 = torch.arange(end = 8, dtype = torch.float32).reshape(1, 2, 4)
    //    val input = torch.randn(Seq(16, 3, 8, 320, 480))
    println(m123(input13).shape) // ArraySeq(1, 2, 7))  Success

    println(m12(input13).shape) // ArraySeq(1, 2, 7))  Success

  }
}

class PaddingSuite
