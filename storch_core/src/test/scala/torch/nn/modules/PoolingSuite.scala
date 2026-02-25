/*
 * Copyright 2022 storch.dev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package torch.nn.modules

import org.bytedeco.javacpp.{DoublePointer, LongPointer}
import org.bytedeco.pytorch.{ConstantPad2dImpl, ConstantPad2dOptions, LPPool3dImpl, LPPool3dOptions}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn
import torch.nn.modules.attention.PositionalEncoding
import torch.nn.modules.attention.Transformer.TransformerActivation.kGELU
import torch.nn.modules.pooling.AdaptiveAvgPool2d
import torch.nn.modules.regularization.Upsample

class LPPool3dRawSuite2 extends munit.FunSuite {
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

//java.lang.NullPointerException: Pointer address of argument 0 is NULL.
//https://pytorch.org/docs/2.0/generated/torch.nn.AvgPool3d.html
class AvgPool3dSuite extends munit.FunSuite {
  test("AvgPool3dSuite output shapes") {
    //    val m12 = nn.AvgPool3d(kernel_size = (3, 2), stride = Some((2, 1)),count_include_pad = true , output_size = (2,3))
    val input = torch.randn(Seq(20, 16, 50, 44, 31))
    //    assertEquals(m12(input).shape, Seq(1, 64, 5, 7))
    //  java.lang.RuntimeException: integer out of range
    val m22 = nn.AvgPool3d(kernel_size = (3, 2, 2), stride = (2, 1, 2), count_include_pad = true)
    assertEquals(m22(input).shape, Seq(20, 16, 24, 43, 15))
  }
}

//java.lang.NullPointerException: Pointer address of argument 0 is NULL.
//https://pytorch.org/docs/2.5/generated/torch.nn.AvgPool2d.html
class AvgPool2dSuite extends munit.FunSuite {
  test("AvgPool2dSuite output shapes") {
    val m12 = nn.AvgPool2d(kernel_size = (3, 2), stride = (2, 1), count_include_pad = true)
    val input1 = torch.randn(Seq(20, 16, 50, 32))
    println(m12(input1).shape)
    assertEquals(m12(input1).shape, Seq(20, 16, 24, 31))

    val input =
      torch.randn(Seq(20, 16, 50, 32)) // java.lang.RuntimeException: stride should not be zero
    val m22 = nn.AvgPool2d(kernel_size = 3, stride = (2), count_include_pad = true)
    println(m22(input).shape)
    assertEquals(m22(input).shape, Seq(20, 16, 24, 15))
  }
}

//java.lang.NullPointerException: Pointer address of argument 0 is NULL.
//
//at
//org.bytedeco.pytorch.LongOptional.put(Native Method)
//at torch
//.nn.modules.pooling.AvgPool1d.< init >(AvgPool1d.scala: 49)
//at torch
//.nn.modules.pooling.AvgPool1d$.apply(AvgPool1d.scala: 75)
//at torch
//.nn.modules.AvgPool1dSuite.$init$$$anonfun$25(PoolingSuite.scala: 471)
//https://pytorch.org/docs/2.0/generated/torch.nn.AvgPool1d.html
class AvgPool1dSuite extends munit.FunSuite {
  test("AvgPool1dSuite output shapes") {
    // java.lang.RuntimeException: pad should be at most half of effective kernel size, but got pad=6878249410449466734, kernel_size=8457542647859737697 and dilation=1
    //    val m12 = nn.AvgPool1d(kernel_size = (3, 2), stride = Some((2, 1)),output_size = (2,3))
    //    val input = torch.randn(Seq(20, 16, 50, 32))
    //    assertEquals(m12(input).shape, Seq(1, 64, 5, 7))

    val m22 = nn.AvgPool1d(kernel_size = 3, stride = (2), count_include_pad = true)
    val input1 = torch.Tensor(Seq(1.0, 2, 3, 4, 5, 6, 7)).reshape(1, 1, 7)
    println(m22(input1.to(torch.float32)).shape)
    //    assertEquals(m22(input1.to(torch.float32)).shape, Seq(1,1,3)) //会出现偏移 ArraySeq(1, 1, 2)
  }
}

//https://pytorch.org/docs/2.3/generated/torch.nn.LPPool1d.html
class LPPool1dSuite extends munit.FunSuite {
  test("AdapativeMaxPool2d output shapes") {
    System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
    val m12 = nn.LPPool1d(norm_type = 2f, kernel_size = 3, stride = (2))
    val input = torch.randn(Seq(20, 16, 50))
    println(m12(input).shape)
    //    assertEquals(m12(input).shape, Seq(20,16,24))

  }
}

// java.lang.ClassCastException: class org.bytedeco.javacpp.BoolPointer
// cannot be cast to class scala.runtime.Nothing$ (org.bytedeco.javacpp.BoolPointer and scala.runtime.Nothing$ are in unnamed module of loader 'app')
//https://pytorch.org/docs/2.0/generated/torch.nn.LPPool2d.html
class LPPool2dSuite extends munit.FunSuite {
  test("AdapativeMaxPool2d output shapes") {
    System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
    // Process finished with exit code -1073740940 (0xC0000374)
    // java.lang.RuntimeException: kernel size should be greater than zero, but got kH: 2 kW: 0
    val m12 = nn.LPPool2d(norm_type = 1.2f, kernel_size = (3, 3), stride = (2, 1), ceil_mode = true)
    // option ceil mode true norm type 1.2 stride 2 1 kernel size 3  0
    // option ceil mode false norm type 1.2 stride 2 1 kernel size 3  0
    val native = m12.nativeModule
    val options = m12.nativeModule.options()
    println(s"option ceil mode ${m12.nativeModule.options().ceil_mode().get()} norm type ${m12.nativeModule
        .options()
        .norm_type()
        .get()} stride ${m12.nativeModule.options().stride().get(0)} ${m12.nativeModule.options().stride().get(1)} kernel size ${m12.nativeModule
        .options()
        .kernel_size()
        .get(0)}  ${m12.nativeModule.options().kernel_size().get(1)}")
    println(
      m12.nativeModule.toString
    ) // option ceil mode true norm type 1.2 stride 2 1 kernel size 3  3
    //    ArraySeq(20, 16, 25, 30)
    val input = torch.randn(Seq(20, 16, 50, 32))
    println(m12(input).shape) // ArraySeq(20, 16, 24, 43, 15)
    //        assertEquals(m12(input).shape, Seq(1, 64, 5, 7))
    //    val m12 = nn.LPPool2d(kernel_size=(7, 7),norm_type=2,stride = Some(1))
    //    val input = torch.randn(Seq(1, 64, 8, 9))
    assertEquals(m12(input).shape, Seq(20, 16, 25, 30))
    //    val m22 = nn.LPPool2d(kernel_size =(1, 1),norm_type=2,stride = Some(1))
    //    assertEquals(m22(input).shape, Seq(1, 64, 1, 1))
  }
}

//java.lang.ClassCastException: class org.bytedeco.javacpp.LongPointer  错误的toNative 和Longpointer（）
// cannot be cast to class scala.runtime.Nothing$ (org.bytedeco.javacpp.LongPointer and scala.runtime.Nothing$ are in unnamed module of loader 'app')
//https://pytorch.org/docs/2.3/generated/torch.nn.LPPool3d.html
class LPPool3dSuite extends munit.FunSuite {
  test("AdapativeMaxPool2d output shapes") {
    // java.lang.RuntimeException: integer out of range
    System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
    val m12 = nn.LPPool3d(kernel_size = (3, 2, 2), norm_type = 1.2f, stride = (2, 1, 2))
    val input = torch.randn(Seq(20, 16, 50, 44, 31))
    println(m12(input).shape) // ArraySeq(20, 16, 24, 43, 15)
    //    ArraySeq(20, 16, 24, 43, 15)
    //    assertEquals(m12(input).shape, Seq(1, 64, 5, 7))
    //    val m22 = nn.LPPool3d(kernel_size = (1, 1), norm_type = 2, stride = Some(1))
    //    assertEquals(m22(input).shape, Seq(1, 64, 1, 1))
  }
}

class AdapativeMaxPool1dSuite0 extends munit.FunSuite {
  test("AdapativeMaxPool1d output shapes") {
    val m1 = nn.AdaptiveMaxPool1d(5)
    val input = torch.randn(Seq(1, 64, 8))
    println(
      m1(input).shape
    ) // Failed DefaultCPUAllocator: not enough memory: you tried to allocate 606564406149120 bytes.
    assertEquals(m1(input).shape, Seq(1, 64, 5)) // //succeed pytorch torch.Size([1, 64, 5])
  }
}

//https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool1d.html
class AdapativeMaxPool1dSuite extends munit.FunSuite {
  test("AdapativeMaxPool1d output shapes") {
    val input2 = torch.randn(Seq(1, 64, 10))
    val m2 = nn.AdaptiveMaxPool1d(7)
    assertEquals(m2(input2).shape, Seq(1, 64, 7)) // succeed
    //    # torch.Size([1, 64, 7])
  }
}

class AdapativeAvgPool1dSuite0 extends munit.FunSuite {
  test("AdapativeAvgPool1d output shapes") {
    val m1 = nn.AdaptiveAvgPool1d(5)
    val input = torch.randn(Seq(1, 64, 8))
    println(
      m1(input).shape
    ) // ArraySeq(1, 64, 5) succeed 少一位  //Failed DefaultCPUAllocator: not enough memory: you tried to allocate 606564406149120 bytes.
    assertEquals(m1(input).shape, Seq(1, 64, 5))
  }
}

//https://pytorch.org/docs/main/generated/torch.nn.AdaptiveAvgPool1d.html
class AdapativeAvgPool1dSuite extends munit.FunSuite {
  test("AdapativeAvgPool1d output shapes") {
    val input2 = torch.randn(Seq(1, 64, 10))
    val m2 = nn.AdaptiveAvgPool1d(7)
    assertEquals(m2(input2).shape, Seq(1, 64, 7)) // succeed
  }
}

class AdapativeAvgPool1dSuite1 extends munit.FunSuite {
  test("AdapativeAvgPool1d output shapes") {
    val m1 = nn.AdaptiveAvgPool1d(5)
    val input = torch.randn(Seq(1, 64, 8))
    println(
      m1(input).shape
    ) // ArraySeq(1, 64, 5) succeed 少一位  //Failed DefaultCPUAllocator: not enough memory: you tried to allocate 606564406149120 bytes.
    assertEquals(m1(input).shape, Seq(1, 64, 5))
  }
}

//https://pytorch.org/docs/main/generated/torch.nn.MaxPool1d.html
class MaxPool1dSuite extends munit.FunSuite {
  test("AdapativeMaxPool2d output shapes") {
    val m12 = nn.MaxPool1d(kernel_size = 3, stride = (2))
    val input = torch.randn(Seq(20, 16, 50))
    assertEquals(m12(input).shape, Seq(20, 16, 16)) // success
//    assertEquals(m12(input).shape, Seq(20, 16, 24)) //success
    //    val m22 = nn.MaxPool1d(kernel_size = (1, 1))
    //    assertEquals(m22(input).shape, Seq(1, 64, 1, 1))
  }
}

//java.lang.ClassCastException: class org.bytedeco.javacpp.LongPointer
// cannot be cast to class scala.runtime.Nothing$ (org.bytedeco.javacpp.LongPointer and scala.runtime.Nothing$ are in unnamed module of loader 'app')
class MaxPool2dSuite extends munit.FunSuite {
  test("AdapativeMaxPool2d output shapes") {
    val m12 = nn.MaxPool2d(kernel_size = (3, 2), stride = (2, 1))
    val input = torch.randn(Seq(20, 16, 50, 32))
    assertEquals(m12(input).shape, Seq(20, 16, 24, 31)) // success
    //    val m22 = nn.MaxPool2d(kernel_size = (1, 1))
    //    assertEquals(m22(input).shape, Seq(1, 64, 1, 1))

    val m123 = nn.MaxPool2d(kernel_size = 3, stride = (2))
    val input3 = torch.randn(Seq(20, 16, 50, 32))
    assertEquals(m123(input3).shape, Seq(20, 16, 24, 15)) // success
  }
}

//java.lang.ClassCastException: class org.bytedeco.javacpp.LongPointer cannot be cast to class scala.runtime.Nothing$ (org.bytedeco.javacpp.LongPointer and scala.runtime.Nothing$ are in unnamed module of loader 'app')
//https://pytorch.org/docs/1.10/generated/torch.nn.MaxUnpool2d.html
//java.lang.RuntimeException: elements in indices should be type int64 but got: Float
class MaxUnPool2dSuite extends munit.FunSuite {
  test("AdapativeMaxPool2d output shapes") {
    val pool = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2), return_indices = true)
    val unpool = nn.MaxUnpool2d(kernel_size = (2, 2), stride = (2, 2))
    // java.lang.RuntimeException: Found an invalid max index: 16 (output volumes are of size 4x4
    val input1 =
      torch.Tensor(Seq(1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)).reshape(1, 1, 4, 4)
    val input = torch.randn(Seq(1, 64, 8, 9))
    val output, indices = pool(input1.to(torch.float32))
    println(s"output: ${output.shape} indices: ${indices.shape}")
    val output22 = unpool(output, indices.to(torch.int64), outputSize = Array(1, 1, 5, 5))
    println(output22.shape)
    //    output: ArraySeq(1, 2, 2) indices: ArraySeq(1, 2, 2)
    // java.lang.RuntimeException: Found an invalid max index: 16 (output volumes are of size 4x4
    //    val output2 = unpool(output,indices.to(torch.int64))
    // println(output2.shape)
    // assertEquals(output2.shape, Seq(1, 64, 5, 7))
    //    val input12 =  torch.Tensor(Seq(1.0,2,3,4,5,6, 7,8,9)).reshape(1,1,9)
    //    val output12,indices12 = pool(input12.to(torch.float32))
  }
}

//原来可以 现在又有问题
//java.lang.ClassCastException: class org.bytedeco.javacpp.LongPointer cannot be cast to class scala.runtime.Nothing$ (org.bytedeco.javacpp.LongPointer and scala.runtime.Nothing$ are in unnamed module of loader 'app')
//java.lang.ClassCastException: class org.bytedeco.javacpp.BoolPointer cannot be cast to class scala.runtime.Nothing$ (org.bytedeco.javacpp.BoolPointer and scala.runtime.Nothing$ are in unnamed module of loader 'app')
//https://pytorch.org/docs/stable/generated/torch.nn.MaxUnpool3d.html
class MaxUnPool3dSuite extends munit.FunSuite {
  test("AdapativeMaxPool2d output shapes") {
    val pool = nn.MaxPool3d(kernel_size = (3, 3, 3), stride = (2, 2, 2), return_indices = true)
    val unpool = nn.MaxUnpool3d(kernel_size = (3, 3, 3), stride = (2, 2, 2))
    // java.lang.RuntimeException: Found an invalid max index: 16 (output volumes are of size 4x4
    //    val input1 = torch.Tensor(Seq(1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)).reshape(1, 4, 4)
    val input1 = torch.randn(Seq(20, 16, 51, 33, 15))
    val output, indices = pool(input1.to(torch.float32))
    println(s"output: ${output.shape} indices: ${indices.shape}")
    //    output: ArraySeq(20, 16, 25, 16, 7) indices: ArraySeq(20, 16, 25, 16, 7)
    val output2 = unpool(output, indices.to(torch.int64))
    println(output2.shape)
    // ArraySeq(20, 16, 51, 33, 15)
  }
}

//java.lang.ClassCastException: class org.bytedeco.javacpp.LongPointer cannot be cast to class scala.runtime.Nothing$ (org.bytedeco.javacpp.LongPointer and scala.runtime.Nothing$ are in unnamed module of loader 'app')
//java.lang.RuntimeException: kernel size should be greater than zero, but got kT: 3 kH: 3 kW: 0
//java.lang.ClassCastException: class org.bytedeco.javacpp.BoolPointer cannot be cast to class scala.runtime.Nothing$ (org.bytedeco.javacpp.BoolPointer and scala.runtime.Nothing$ are in unnamed module of loader 'app')
//java.lang.RuntimeException: kernel size should be greater than zero, but got kT: 3 kH: 3 kW: 0
//https://pytorch.org/docs/stable/generated/torch.nn.MaxPool3d.html
class MaxPool3dSuite extends munit.FunSuite {
  test("AdapativeMaxPool2d output shapes") {
    val m12 = nn.MaxPool3d(kernel_size = (3, 2, 2), stride = (2, 1, 2))
    val input = torch.randn(Seq(20, 16, 50, 44, 31))
    assertEquals(m12(input).shape, Seq(20, 16, 24, 43, 15)) // success

//    val m123 = nn.MaxPool3d(kernel_size =3, stride = Some((2, 1, 2)))
//    val input3 = torch.randn(Seq(20, 16, 50, 44, 31))
//    assertEquals(m123(input3).shape, Seq(20, 16, 24, 43, 15))  //failed

  }
}

//java.lang.RuntimeException: Found an invalid max index
//: 8(output volumes are of size
//  8 x1
//    Exception raised from cpu_max_unpool at D :/ a / javacpp -presets / javacpp - presets / pytorch / cppbuild / windows - x86_64 - gpu / pytorch / aten / src / ATen / native / cpu / MaxUnpoolKernel.cpp: 89
// Found an invalid max index: 4 (output volumes are of size 4x1
//https://pytorch.org/docs/2.0/generated/torch.nn.MaxUnpool1d.html
class MaxUnPool1dSuite extends munit.FunSuite {
  test("AdapativeMaxPool2d output shapes") {
    // java.lang.RuntimeException: Found an invalid max index: 8 (output volumes are of size 8x1
    val pool = nn.MaxPool1d(kernel_size = (2, 2), stride = (2, 2), return_indices = true)
    val unpool = nn.MaxUnpool1d(kernel_size = 2, stride = (2))
    val input1 = torch.Tensor(Seq(1.0, 2, 3, 4, 5, 6, 7, 8)).reshape(1, 1, 8)
    val input = torch.randn(Seq(1, 64, 8, 9))
    val output, indices = pool(input1.to(torch.float32))
    println(s"output: ${output.shape} indices: ${indices.shape}")
//    output: ArraySeq
//    (1, 1, 4) indices: ArraySeq
//    (1, 1, 4)
//    output12: ArraySeq
//    (1, 1, 4) indices12: ArraySeq
//    (1, 1, 4)
    //    assertEquals(output2.shape, Seq(1, 64, 5, 7))
    val input12 = torch.Tensor(Seq(1.0, 2, 3, 4, 5, 6, 7, 8)).reshape(1, 1, 8)
    val output12, indices12 = pool(input12.to(torch.float32))
    println(s"output12: ${output12.shape} indices12: ${indices12.shape}")
    // java.lang.RuntimeException: Found an invalid max index: 8 (output volumes are of size 8x1
    //    val output2 = unpool(output, indices.to(torch.int64))
    //    println(output2.shape)

    //    java.lang.RuntimeException: Found an invalid max index
    //    : 8(output volumes are of size
    //    8 x1
    val output22 = unpool(output, indices.to(torch.int64))
    println(output22.shape)
    //    val m22 = nn.MaxUnpool1d(kernel_size = (1, 1))
    //    assertEquals(m22(input, input.to(torch.int64)).shape, Seq(1, 64, 1, 1))
  }
}

//https://pytorch.org/docs/main/generated/torch.nn.FractionalMaxPool3d.html
//java.lang.RuntimeException: FractionalMaxPool2d requires specifying either an output size, or a pooling ratio
class FractionalMaxPool2dSuite extends munit.FunSuite {
  test("AdapativeMaxPool2d output shapes") {
    val m13 = nn.FractionalMaxPool2d(
      kernel_size = (7, 7),
      output_size = Some(7, 7),
      output_ratio = Some(0.57f, 05f)
    )
    val input = torch.randn(Seq(1, 64, 8, 9))
    println(
      s" options kernel ${m13.nativeModule.options().kernel_size().get(0)} k2 ${m13.nativeModule.options().kernel_size().get(1)} outsize ${m13.nativeModule
          .options()
          .output_size()
          .has_value()}  ${m13.nativeModule.options().output_size().getPointer(0)} out2 ${m13.nativeModule
          .options()
          .output_size()
          .getPointer(1)} outRatio ${m13.nativeModule.options().output_ratio().has_value()} ${m13.nativeModule
          .options()
          .output_ratio()
          .getPointer(0)} ratio2 ${m13.nativeModule.options().output_ratio().getPointer(1)}"
    )
    assertEquals(m13(input.to(torch.float64)).shape, Seq(1, 64, 5, 7))
  }
}

class FractionalMaxPool3dSuite extends munit.FunSuite {
  test("FractionalMaxPool3dSuite output shapes") {
    val input = torch.randn(Seq(1, 64, 8, 9))
    val m23 = nn.FractionalMaxPool3d(
      kernel_size = (4, 8, 1),
      output_size = Some(5, 6, 7),
      output_ratio = Some(0.4f, 0.34f, 0.57f)
    )
    println(
      s" options kernel ${m23.nativeModule.options().kernel_size().get(0)} k2 ${m23.nativeModule.options().kernel_size().get(1)} outsize ${m23.nativeModule
          .options()
          .output_size()
          .has_value()}  ${m23.nativeModule.options().output_size().getPointer(0)} out2 ${m23.nativeModule
          .options()
          .output_size()
          .getPointer(1)} outRatio ${m23.nativeModule.options().output_ratio().has_value()} ${m23.nativeModule
          .options()
          .output_ratio()
          .getPointer(0)} ratio2 ${m23.nativeModule.options().output_ratio().getPointer(1)}"
    )
    println(m23(input.to(torch.float64)).shape)

  }
}

class AdapativeAvgPool2dSuite0 extends munit.FunSuite {
  test("AdapativeAvgPool2d output shapes") {
    val m1 = new AdaptiveAvgPool2d((5, 7))
    val input = torch.randn(Seq(1, 64, 8, 9))
    assertEquals(
      m1(input).shape,
      Seq(1, 64, 5, 7)
    ) // ArraySeq(1, 64, 5, 217) 只有 tuple2 正常工作  ArraySeq(1, 64, 5, 0) 多个零
//     java.lang.RuntimeException: Stride calculation overflowed
//pytorch torch.Size([1, 64, 5, 7])
  }
}

//https://pytorch.org/docs/main/generated/torch.nn.AdaptiveAvgPool2d.html
class AdapativeAvgPool2dSuite1 extends munit.FunSuite {
  test("AdapativeAvgPool2d output shapes") {
    val input2 = torch.randn(Seq(1, 64, 10, 9))
    val m2 = nn.AdaptiveAvgPool2d(7)
    assertEquals(m2(input2).shape, Seq(1, 64, 7, 7)) // Actual   :ArraySeq(1, 64, 7, 0)  多个零
//java.lang.RuntimeException: Stride calculation overflowed
  }
}
//java.lang.NullPointerException: Pointer address of argument 1 is NULL.
class AdapativeAvgPool2dSuite2 extends munit.FunSuite {
  test("AdapativeAvgPool2d output shapes") {
    val input2 = torch.randn(Seq(1, 64, 10, 9))
    val m3 = nn.AdaptiveAvgPool2d((None, 7))
    assertEquals(m3(input2).shape, Seq(1, 64, 10, 7))
    // pytorchtorch.     # torch.Size([1, 64, 10, 7])
  }
}

//#target output size of 5 x7x9
//  m = nn.AdaptiveAvgPool3d((5, 7, 9))
//input = torch.randn(1, 64, 8, 9, 10)
//output = m(input)
//#target output size of 7 x7x7 (cube)
//m = nn.AdaptiveAvgPool3d(7)
//input = torch.randn(1, 64, 10, 9, 8)
//output = m(input)
//#target output size of 7 x9x8
//  m = nn.AdaptiveAvgPool3d((7, None, None))
//input = torch.randn(1, 64, 10, 9, 8)
//output = m(input)
//torch.Size([1, 64, 5, 7, 9])
class AdapativeAvgPool3dSuite1 extends munit.FunSuite {
  test("AdapativeAvgPool2d output shapes") {
    val m1 = nn.AdaptiveAvgPool3d((5, 7, 9))
    val input = torch.randn(Seq(1, 64, 8, 9, 10))
    val output = m1(input)
    // java.lang.RuntimeException: Stride calculation overflowed
    assertEquals(
      m1(input).shape,
      Seq(1, 64, 5, 7, 9)
    ) // ArraySeq(1, 64, 5, 9, 0)  ArraySeq(1, 64, 5, 9, 10) 多个零 | pytorch torch.Size([1, 64, 5, 7, 9])
  }
}
//https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool3d.html
class AdapativeAvgPool3dSuite2 extends munit.FunSuite {
  test("AdapativeAvgPool2d output shapes") {
    // java.lang.RuntimeException: Stride calculation overflowed
    val m2 = nn.AdaptiveAvgPool3d((7))
    val input2 = torch.randn(Seq(1, 64, 10, 9, 8))
    val output2 = m2(input2)
    assertEquals(
      output2.shape,
      Seq(1, 64, 7, 7, 7)
    ) // ArraySeq(1, 64, 7, 7, 0) 多个零 |pytorch torch.Size([1, 64, 7, 7, 7])
  }
}

class AdapativeAvgPool3dSuite31 extends munit.FunSuite {
  test("AdapativeAvgPool2d output shapes") {
// pytorch torch.Size([1, 64, 7, 9, 8])
//        val m3 = nn.AdaptiveAvgPool3d((7,None,None))
//        val input3 = torch.randn(Seq(1, 64, 10, 9, 8))
//        val output3 = m2(input3)
//        println(output2.shape)
//        assertEquals(m2(input).shape, Seq(1, 64, 7, 9,8))
  }
}

//#target output size of 5 x7
//  m = nn.AdaptiveAvgPool2d((5, 7))
//input = torch.randn(1, 64, 8, 9)
//output = m(input)
//#target output size of 7 x7 (square)
//m = nn.AdaptiveAvgPool2d(7)
//input = torch.randn(1, 64, 10, 9)
//output = m(input)
//#target output size of 10 x7
//  m = nn.AdaptiveAvgPool2d((None, 7))
//input = torch.randn(1, 64, 10, 9)
//output = m(input)

//https://pytorch.org/docs/2.2/generated/torch.nn.AdaptiveMaxPool2d.html
class AdapativeMaxPool2dSuite3 extends munit.FunSuite {
  test("AdapativeMaxPool2d output shapes") {
    // java.lang.RuntimeException: Storage size calculation overflowed with sizes=[1, 64, 1912110652560, 1912110652592]
    val m1 = nn.AdaptiveMaxPool2d((5, 7))
    val input = torch.randn(Seq(1, 64, 8, 9))
    assertEquals(
      m1(input).shape,
      Seq(1, 64, 5, 7)
    ) // ArraySeq(1, 64, 5, 0) 多个零  |pytorch torch.Size([1, 64, 5, 7])

  }
}

class AdapativeMaxPool2dSuite4 extends munit.FunSuite {
  test("AdapativeMaxPool2d output shapes") {
    val input2 = torch.randn(Seq(1, 64, 10, 9))
    val m2 = nn.AdaptiveMaxPool2d((7))
    assertEquals(
      m2(input2).shape,
      Seq(1, 64, 7, 7)
    ) // Actual   :ArraySeq(1, 64, 7, 0)  pytorch torch.Size([1, 64, 7, 7])
  }
}
class AdapativeMaxPool2dSuite41 extends munit.FunSuite {
  test("AdapativeMaxPool2d output shapes") {
// pytorch torch.Size([1, 64, 10, 7])
    val input2 = torch.randn(Seq(1, 64, 10, 9))
    val m3 = nn.AdaptiveAvgPool2d((None, 7))
    assertEquals(m3(input2).shape, Seq(1, 64, 10, 7))

  }
}

class AdapativeMaxPool3dSuite5 extends munit.FunSuite {
  test("AdapativeAvgPool2d output shapes") {
    val m1 = nn.AdaptiveMaxPool3d((5, 7, 9))
    val input = torch.randn(Seq(1, 64, 8, 9, 10))
    val output = m1(input)
    assertEquals(m1(input).shape, Seq(1, 64, 5, 7, 9)) // ArraySeq(1, 64, 5, 9, 0) 多个零
    // java.lang.RuntimeException: Storage size calculation overflowed with sizes=[1, 64, 5, 9, 7021802828931744845]
  }
}
// https://pytorch.org/docs/2.4/generated/torch.nn.AdaptiveMaxPool3d.html
class AdapativeMaxPool3dSuite4 extends munit.FunSuite {
  test("AdapativeAvgPool2d output shapes") {

    val m2 = nn.AdaptiveMaxPool3d((7, 7, 7))
    val input2 = torch.randn(Seq(1, 64, 10, 9, 8))
    val output2 = m2(input2)
    println(
      output2.shape
    ) // ArraySeq(1, 64, 7, 7, 0)  ArraySeq(1, 64, 7, 7, 8)  pytorch torch.Size([1, 64, 7, 7, 7])
    assertEquals(m2(input2).shape, Seq(1, 64, 7, 7, 7))

  }
}

class AdapativeMaxPool3dSuite6 extends munit.FunSuite {
  test("AdapativeAvgPool2d output shapes") {
//// pytorch  torch.Size([1, 64, 7, 9, 8])
//        val m3 = nn.AdaptiveMaxPool3d((7,None,None))
//        val input3 = torch.randn(Seq(1, 64, 10, 9, 8))
//        val output3 = m2(input3)
//        println(output2.shape)
//        assertEquals(m2(input).shape, Seq(1, 64, 7,9,8))
  }
}

//java.lang.RuntimeException: Storage size calculation overflowed with sizes=[1, 64, 1390164960288, 1390164960320]
class AdapativeMaxPool2dSuite extends munit.FunSuite {
  test("AdapativeMaxPool2d output shapes") {
    val m12 = nn.AdaptiveMaxPool2d((5, 7))
    val input = torch.randn(Seq(1, 64, 8, 9))
    assertEquals(m12(input).shape, Seq(1, 64, 5, 7))
    val m22 = nn.AdaptiveMaxPool2d((1, 1))
    assertEquals(m22(input).shape, Seq(1, 64, 1, 1))
  }
}

//Expected: List
//(1, 64, 5, 7)
//Actual: ArraySeq
//(1, 64, 5, 0)

class pad2dSuite extends munit.FunSuite {
  test("AdapativeMaxPool2d output shapes") {
    val m12 = nn.ConstantPad2d(padding = 7, value = 4f)
    val input = torch.randn(Seq(1, 64, 8, 9))
    assertEquals(m12(input).shape, Seq(1, 64, 22, 23))
    val m22 = nn.ReflectionPad2d(padding = 7)
    assertEquals(m22(input).shape, Seq(1, 64, 22, 23))
  }
}

//    val m1 = nn.TransformerEncoderLayer(d_model = 8,n_head = 4)
//    val m12 = nn.MultiheadAttention(embed_dim = 4,num_heads = 8,dropout = 0.1,kdim = 8,bias = true,vdim = 8)
// Process finished with exit code -1073740940 (0xC0000374)
//    val m22 = PositionalEncoding(d_model = 8)
//    assertEquals(m22(input).shape, Seq(1, 64, 1, 1))
//    val m1 = nn.TransformerEncoderLayer(d_model = 8,n_head = 4)
//    val m12 = nn.MultiheadAttention(embed_dim = 4,num_heads = 8,dropout = 0.1,kdim = 8,bias = true,vdim = 8)

//    assertEquals(m12(input).shape, Seq(1, 64, 5, 7)) //lack of  paramter
//    println(res.shape) // Process finished with exit code -1073740940 (0xC0000374)
//    val m22 = PositionalEncoding(d_model = 8)
//    assertEquals(m22(input).shape, Seq(1, 64, 1, 1))

//    raw options kernel
//    7 k2 7 outsize true
//    7 out2 7598807741461061480 outRatio true
//    0.5699999928474426 ratio2 1.9108424296605356E214
//    //    , outputSize: Some
//    //    ((7, 7)) outputRatio Some((0.57, 5.0))
//    //    raw options kernel
//    //    7 k2 7 outsize true
//    //    7 out2 7598807741461061480 outRatio true
//    //    0.5699999928474426 ratio2 1.9108424296605356E214
//
//    //    assertEquals(m23(input.to(torch.float64)).shape, Seq(1, 64, 1, 1))
//    val m23 = nn.FractionalMaxPool3d(kernel_size = (4, 8, 1), output_size = Some(5), output_ratio =  Some(0.57f))
//    val m23 = nn.FractionalMaxPool3d(kernel_size = (4, 8, 1), output_size = Some(5, 6, 7), output_ratio =  Some(0.4f, 0.34f, 0.57f))
//    println(s" options kernel ${m23.nativeModule.options().kernel_size().get(0)} k2 ${m23.nativeModule.options().kernel_size().get(1)} outsize ${m23.nativeModule.options().output_size().has_value()}  ${m23.nativeModule.options().output_size().getPointer(0)} out2 ${m23.nativeModule.options().output_size().getPointer(1)} outRatio ${m23.nativeModule.options().output_ratio().has_value()} ${m23.nativeModule.options().output_ratio().getPointer(0)} ratio2 ${m23.nativeModule.options().output_ratio().getPointer(1)}")
//    randomSamples is None
//    , outputSize: Some
//    ((7, 7)) outputRatio Some((0.57, 5.0))
//    raw options kernel
//    7 k2 7 outsize true
//    7 out2 7598807741461061480 outRatio true
//    0.5699999928474426 ratio2 1.9108424296605356E214

//    assertEquals(m23(input.to(torch.float64)).shape, Seq(1, 64, 1, 1))
//    val m13 = nn.FractionalMaxPool2d(kernel_size = (7, 7), output_size = Some(7, 7), output_ratio = Some(0.57f, 05f))
//    raw options kernel
//    7 k2 7 outsize true
//    7 out2 7598807741461061480 outRatio true
//    0.5699999928474426 ratio2 1.9108424296605356E214
//    println(s" options kernel ${m13.nativeModule.options().kernel_size().get(0)} k2 ${m13.nativeModule.options().kernel_size().get(1)} outsize ${m13.nativeModule.options().output_size().has_value()}  ${m13.nativeModule.options().output_size().getPointer(0)} out2 ${m13.nativeModule.options().output_size().getPointer(1)} outRatio ${m13.nativeModule.options().output_ratio().has_value()} ${m13.nativeModule.options().output_ratio().getPointer(0)} ratio2 ${m13.nativeModule.options().output_ratio().getPointer(1)}")
//    assertEquals(m13(input.to(torch.float64)).shape, Seq(1, 64, 5, 7))
//        val m23 = nn.FractionalMaxPool3d(kernel_size = (4, 8, 1), output_size = Some(5), output_ratio =  Some(0.57f))
