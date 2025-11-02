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

import org.bytedeco.javacpp.{BoolPointer, DoublePointer, FloatPointer, LongPointer}
import org.bytedeco.pytorch.*
import torch.internal.NativeConverters.fromNative
import torch.{Tensor, nn}
import torch.nn.modules.batchnorm.{BatchNorm1d, BatchNorm2d, BatchNorm3d}

class InstanceNorm2dSuite1 extends munit.FunSuite {
  test("InstanceNorm2dSuite1 output shapes") {
    val m12 = nn.InstanceNorm2d(num_features = 100, affine = true)
    val input1 = torch.randn(Seq(20, 100, 35, 45))
    //    val input2 = torch.randn(Seq(16, 5, 7, 7, 7, 7))
    assertEquals(m12(input1).shape, Seq(20, 100, 35, 45)) // torch.Size([20, 100,35,45])
    //    println(m12(input))
  }
}

//java.lang.RuntimeException: weight should contain 20 elements not 100
class InstanceNorm3dSuite extends munit.FunSuite {
  test("InstanceNorm3dSuite output shapes") {
    System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
    val m12 = nn.InstanceNorm3d(num_features = 100, affine = false)
    val input1 = torch.randn(Seq(20, 100, 35, 45, 10))
    //    val input2 = torch.randn(Seq(16, 5, 7, 7, 7, 7))
    assertEquals(m12(input1).shape, Seq(20, 100, 35, 45, 10)) // torch.Size([20, 100,35,45])
    //    println(m12(input))
  }
}

class InstanceNorm3dRawSuite extends munit.FunSuite {
  test("InstanceNorm3dRawSuite output shapes") {
    System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
    val input = torch.randn(Seq(20, 100, 35, 45, 10))
    val numFeatures: Int = 100
    val eps: Double = 1e-05
    val momentum: Float | Option[Float] = 0.1f
    val affine: Boolean = false
    val trackRunningStats: Boolean = true
    val options: InstanceNormOptions = InstanceNormOptions(LongPointer(1).put(numFeatures.toLong))
    options.eps().put(DoublePointer(1).put(eps))
    options.affine().put(affine) // BoolPointer(if affine then 1 else 0))
    momentum match {
      case m: Float => options.momentum().put(DoublePointer(1).put(m.toDouble))
      case m: Option[Float] =>
        if m.isDefined then options.momentum().put(DoublePointer(1).put(m.get.toDouble))
    }

    options.num_features().put(LongPointer(1).put(numFeatures))
    options.track_running_stats.put(trackRunningStats) // if trackRunningStats then 1 else 0))
    val nativeModule: InstanceNorm3dImpl = InstanceNorm3dImpl(options)
    val output = fromNative(nativeModule.forward(input.native))
    println(s"output shape: ${output.shape}")
    //    val m12 = nn.InstanceNorm3d(num_features = 100, affine = false)
    //    val input1 = torch.randn(Seq(20, 100, 35, 45, 10))
    //    //    val input2 = torch.randn(Seq(16, 5, 7, 7, 7, 7))
    //    assertEquals(m12(input1).shape, Seq(20, 100, 35, 45, 10)) //torch.Size([20, 100,35,45])
    //    //    println(m12(input))
    //
    //    val batchNorm = nn.InstanceNorm2d(100, affine = false)
    //    val input = torch.randn(Seq(20, 100, 35, 45))
    //    val output = batchNorm(input)
    //    println(s"output shape: ${output.shape}") //output shape: ArraySeq(20, 100, 35, 45)

  }
}

class InstanceNorm2dRawSuite extends munit.FunSuite {
  test("InstanceNorm2dRawSuite output shapes") {
    System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
    val input = torch.randn(Seq(20, 100, 35, 45))
    val numFeatures: Int = 100
    val eps: Double = 1e-05
    val momentum: Float | Option[Float] = 0.1f
    val affine: Boolean = false
    val trackRunningStats: Boolean = true
    val options: InstanceNormOptions = InstanceNormOptions(LongPointer(1).put(numFeatures.toLong))
    options.eps().put(DoublePointer(1).put(eps))
    options.affine().put(affine) // BoolPointer(if affine then 1 else 0))
    momentum match {
      case m: Float => options.momentum().put(DoublePointer(1).put(m.toDouble))
      case m: Option[Float] =>
        if m.isDefined then options.momentum().put(DoublePointer(1).put(m.get.toDouble))
    }

    options.num_features().put(LongPointer(1).put(numFeatures))
    options.track_running_stats.put(trackRunningStats) // if trackRunningStats then 1 else 0))
    val nativeModule: InstanceNorm2dImpl = InstanceNorm2dImpl(options)
    val output = fromNative(nativeModule.forward(input.native))
    //    val m12 = nn.InstanceNorm3d(num_features = 100, affine = false)
    //    val input1 = torch.randn(Seq(20, 100, 35, 45, 10))
    //    //    val input2 = torch.randn(Seq(16, 5, 7, 7, 7, 7))
    //    assertEquals(m12(input1).shape, Seq(20, 100, 35, 45, 10)) //torch.Size([20, 100,35,45])
    //    //    println(m12(input))
    //
    //    val batchNorm = nn.InstanceNorm2d(100, affine = false)
    //    val input = torch.randn(Seq(20, 100, 35, 45))
    //    val output = batchNorm(input)
    println(s"output shape: ${output.shape}") // output shape: ArraySeq(20, 100, 35, 45)

  }
}

class InstanceNorm2dSuite extends munit.FunSuite {
  test("InstanceNorm2dSuite output shapes") {
    System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
    val m12 = nn.InstanceNorm3d(num_features = 100, affine = false)
    val input1 = torch.randn(Seq(20, 100, 35, 45, 10))
    //    val input2 = torch.randn(Seq(16, 5, 7, 7, 7, 7))
    assertEquals(m12(input1).shape, Seq(20, 100, 35, 45, 10)) // torch.Size([20, 100,35,45])
    //    println(m12(input))

    val batchNorm = nn.InstanceNorm2d(100, affine = false)
    val input = torch.randn(Seq(20, 100, 35, 45))
    val output = batchNorm(input)
    println(s"output shape: ${output.shape}") // output shape: ArraySeq(20, 100, 35, 45)
  }
}

class InstanceNorm1dSuite extends munit.FunSuite {
  test("InstanceNorm1dSuite output shapes") {
    System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
    val batchNorm = nn.InstanceNorm1d(100, eps = 1e-05f)
    val input = torch.randn(Seq(20, 100))
    val output = batchNorm(input)
    println(s"output shape: ${output.shape}")
  }
}

//java.lang.NoSuchMethodError: 'double torch.nn.modules.batchnorm.BatchNorm3d$.apply$default$2()
class batchNorm3dSuite extends munit.FunSuite {
  test("batchNorm3dSuite output shapes") {
    System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
    val batchNorm = nn.BatchNorm3d(100, affine = false, eps = 1e-05f)
    val input = torch.randn(Seq(20, 100, 35, 45, 10))
    val output = batchNorm(input)
    println(s"output shape: ${output.shape}") // output shape: ArraySeq(20, 100, 35, 45, 10)
  }
}

class batchNorm2dSuite extends munit.FunSuite {
  test("batchNorm2dSuite output shapes") {
    System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
    val batchNorm = nn.BatchNorm2d(100, affine = false)
    val input = torch.randn(Seq(20, 100, 35, 45))
    val output = batchNorm(input)
    println(s"output shape: ${output.shape}") // output shape: ArraySeq(20, 100, 35, 45)
  }
}

class batchNorm1dSuite extends munit.FunSuite {
  test("batchNorm1dSuite output shapes") {
    System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
    val batchNorm = nn.BatchNorm1d(100)
    val input = torch.randn(Seq(20, 100))
    val output = batchNorm(input)
    println(s"output shape: ${output.shape}")
  }
}

class batchNorm1dRawSuite extends munit.FunSuite {
  test("batchNorm1dRawSuite output shapes") {
    System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
    val numFeatures: Int = 100
    val eps: Double = 1e-05
    val momentum: Float | Option[Float] = 0.1f
    val affine: Boolean = true
    val trackRunningStats: Boolean = true
    println(s"numFeatures: ${numFeatures}")
    val options = new BatchNormOptions(LongPointer(1).put(numFeatures))
    options.eps().put(DoublePointer(1).put(eps))
    momentum match {
      case m: Float => options.momentum().put(DoublePointer(1).put(m.toDouble))
      case m: Option[Float] =>
        if m.isDefined then options.momentum().put(DoublePointer(1).put(m.get.toDouble))
    }
    //  options.momentum().put(DoublePointer(momentum))
    options.affine().put(affine) // BoolPointer(if affine then 1 else 0))
    options
      .track_running_stats()
      .put(trackRunningStats) // BoolPointer(if trackRunningStats then 1 else 0))
    println("options: ${options}")
    val nativeModule: BatchNorm1dImpl = BatchNorm1dImpl(options)
    val input = torch.randn(Seq(20, 100)).native
    val output = fromNative(nativeModule.forward(input))
    println(s"output shape: ${output.shape}")

  }
}

class BatchNormSuite extends munit.FunSuite {

  def main(args: Array[String]): Unit = {

    println("BatchNormSuite")
    bbatchNormSuite1d()
  }

  def bbatchNormSuite1d(): Unit = {
    val numFeatures: Int = 100
    val eps: Double = 1e-05
    val momentum: Float | Option[Float] = 0.1f
    val affine: Boolean = true
    val trackRunningStats: Boolean = true
    println(s"numFeatures: ${numFeatures}")
    val options = new BatchNormOptions(LongPointer(1).put(numFeatures))
    options.eps().put(DoublePointer(1).put(eps))
    momentum match {
      case m: Float => options.momentum().put(DoublePointer(1).put(m.toDouble))
      case m: Option[Float] =>
        if m.isDefined then options.momentum().put(DoublePointer(1).put(m.get.toDouble))
    }
    //  options.momentum().put(DoublePointer(momentum))
    options.affine().put(affine) // BoolPointer(if affine then 1 else 0))
    options
      .track_running_stats()
      .put(trackRunningStats) // BoolPointer(if trackRunningStats then 1 else 0))
    println("options: ${options}")
    val nativeModule: BatchNorm1dImpl = BatchNorm1dImpl(options)
    val input = torch.randn(Seq(20, 100)).native
    val output = fromNative(nativeModule.forward(input))
    println(s"output shape: ${output.shape}")
  }
  test("BatchNorm3d") {
    torch.manualSeed(0)
    System.setProperty("org.bytedeco.javacpp.logger.debug", "true")
    val m = BatchNorm3d(num_features = 3)
    val input = torch.randn(Seq(3, 3, 10, 10, 3))
    val output = m(input)
    assertEquals(output.shape, input.shape)
    val expectedOutput = Tensor(
      Seq(
        Seq(1.4014f, -0.1438f, -1.2519f),
        Seq(-0.5362f, -1.1465f, 0.0564f),
        Seq(-0.8651f, 1.2903f, 1.1956f)
      )
    )
    assert(torch.allclose(output, expectedOutput, atol = 1e-4))
  }
  test("BatchNorm2d") {
    torch.manualSeed(0)
    System.setProperty("org.bytedeco.javacpp.logger.debug", "true")
    val m = BatchNorm2d(num_features = 3)
    val input = torch.randn(Seq(3, 3, 10, 10))
    val output = m(input)
    assertEquals(output.shape, input.shape)
    val expectedOutput = Tensor(
      Seq(
        Seq(1.4014f, -0.1438f, -1.2519f),
        Seq(-0.5362f, -1.1465f, 0.0564f),
        Seq(-0.8651f, 1.2903f, 1.1956f)
      )
    )
    assert(torch.allclose(output, expectedOutput, atol = 1e-4))
  }
  test("BatchNorm1d") {
    torch.manualSeed(0)
    System.setProperty("org.bytedeco.javacpp.logger.debug", "true")
    val m = BatchNorm1d(num_features = 3)
    val input = torch.randn(Seq(3, 3))
    val output = m(input)
    assertEquals(output.shape, input.shape)
    val expectedOutput = Tensor(
      Seq(
        Seq(1.4014f, -0.1438f, -1.2519f),
        Seq(-0.5362f, -1.1465f, 0.0564f),
        Seq(-0.8651f, 1.2903f, 1.1956f)
      )
    )
    assert(torch.allclose(output, expectedOutput, atol = 1e-4))
  }

  test("BatchNorm2d") {
    torch.manualSeed(0)
    val m = BatchNorm2d(num_features = 3)
    val input = torch.randn(Seq(3, 3, 1, 1))
    val output = m(input)
    assertEquals(output.shape, input.shape)
    val expectedOutput = Tensor(
      Seq(
        Seq(1.4014f, -0.1438f, -1.2519f),
        Seq(-0.5362f, -1.1465f, 0.0564f),
        Seq(-0.8651f, 1.2903f, 1.1956f)
      )
    )
    assert(torch.allclose(output.squeeze, expectedOutput, atol = 1e-4))
  }
}
