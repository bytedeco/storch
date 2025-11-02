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

package torch.nn

import functional as F

import torch.{ComplexNN, FloatNN, nn}

class Conv1dSuite extends munit.FunSuite {
  test("Conv1dSuite output shapes") {
    val m12 = nn.Conv1d(in_channels = 16, out_channels = 33, kernel_size = 3, stride = 2)
    val input1 = torch.randn(Seq(20, 16, 50))

    //    val input2 = torch.randn(Seq(16, 5, 7, 7, 7, 7))
    assertEquals(m12(input1).shape, Seq(20, 33, 24)) // torch.Size([100])
    //    println(m12(input))
  }
}

class Conv2dSuite1 extends munit.FunSuite {
  test("Conv2dSuite1 output shapes") {
    val m12 = nn.Conv2d(
      in_channels = 16,
      out_channels = 33,
      kernel_size = (3, 5),
      stride = (2, 1),
      padding = (4, 2),
      dilation = (3, 1)
    )
    val input1 = torch.randn(Seq(20, 16, 50, 100))
    val output = m12(input1)
    //    val input2 = torch.randn(Seq(16, 5, 7, 7, 7, 7))
    assertEquals(
      output.shape,
      Seq(20, 33, 26, 100)
    ) // torch.Size([100])  failed Actual   :ArraySeq(20, 33, 22, 96)
    //    println(m12(input))
  }
}

//Conv2d(16, 33, kernel_size=(3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
//input = torch.randn(20, 16, 50, 100)
//torch.Size([20, 33, 26, 100])
//java.lang.ClassCastException: class org.bytedeco.javacpp.LongPointer cannot be cast to class scala.runtime.Nothing$ (org.bytedeco.javacpp.LongPointer and scala.runtime.Nothing$ are in unnamed module of loader 'app')
class Conv2dSuite extends munit.FunSuite {
  test("Conv2dSuite output shapes") {
    val m12 = nn.Conv2d(
      in_channels = 16,
      out_channels = 33,
      kernel_size = (3, 5),
      stride = (2, 1),
      dilation = (3, 1)
    )
    val input1 = torch.randn(Seq(20, 16, 50, 100))
    val output = m12(input1)
    //    val input2 = torch.randn(Seq(16, 5, 7, 7, 7, 7))
    assertEquals(
      output.shape,
      Seq(20, 33, 22, 96)
    ) // torch.Size([100])  failed Actual   :ArraySeq(20, 33, 22, 96)
    //    println(m12(input))
  }
}

//java.lang.ClassCastException: class org.bytedeco.javacpp.LongPointer cannot be cast to class scala.runtime.Nothing$ (org.bytedeco.javacpp.LongPointer and scala.runtime.Nothing$ are in unnamed module of loader 'app')
//有问题  java.lang.RuntimeException: from is out of bounds for float
class Conv3dSuite0 extends munit.FunSuite {
  test("Conv3dSuite0 output shapes") {
    val m12 = nn.Conv3d(in_channels = 16, out_channels = 33, kernel_size = 3, stride = 2)
    val input1 = torch.randn(Seq(20, 16, 10, 50, 100))
    //    val input2 = torch.randn(Seq(16, 5, 7, 7, 7, 7))
    assertEquals(m12(input1).shape, Seq(20, 33, 4, 24, 49)) // torch.Size([100])
    println(m12(input1).shape)
  }
}

//java.lang.ClassCastException: class org.bytedeco.javacpp.LongPointer cannot be cast to class scala.runtime.Nothing$ (org.bytedeco.javacpp.LongPointer and scala.runtime.Nothing$ are in unnamed module of loader 'app')
class Conv3dSuite extends munit.FunSuite {
  test("Conv3dSuite output shapes") {
    val m12 = nn.Conv3d(
      in_channels = 16,
      out_channels = 33,
      kernel_size = (3, 5, 2),
      stride = (2, 1, 1),
      padding = (4, 2, 0)
    )
    val input1 = torch.randn(Seq(20, 16, 10, 50, 100))
    //    val input2 = torch.randn(Seq(16, 5, 7, 7, 7, 7))
    assertEquals(m12(input1).shape, Seq(20, 33, 8, 50, 99)) // torch.Size([100])
    println(m12(input1).shape)
  }
}

class ConvTranspose1dSuite extends munit.FunSuite {
  test("ConvTranspose1d output shapes") {
    val m12 = nn.ConvTranspose1d(
      in_channels = 16,
      out_channels = 32,
      kernel_size = 3,
      stride = 2,
      padding = 1
    ) // (5, 7), padding_mode = "reflect")
    val input = torch.randn(Seq(1, 16, 10)) // , 64, 8, 9))
    val output = m12(input)
    assertEquals(output.shape, Seq(1, 32, 19)) // 20)) //, 64, 5, 7))
  }
}

//java.lang.RuntimeException: Given transposed=1, weight of size [7, 4, 5, 7], expected input[1, 64, 8, 9] to have 7 channels, but got 64 channels instead
class ConvTranspose2dSuite extends munit.FunSuite {
  test("ConvTranspose2d output shapes") {
    val m12 = nn.ConvTranspose2d(
      in_channels = 16,
      out_channels = 33,
      kernel_size = 3,
      stride = 2
    ) // (5, 7), padding_mode = "reflect")
    val input = torch.randn(Seq(20, 16, 50, 100)) // , 64, 8, 9))
    assertEquals(m12(input).shape, Seq(20, 33, 101, 201)) // , 64, 5, 7))
  }
}

class ConvTranspose2dSuite2 extends munit.FunSuite {
  test("ConvTranspose2d output shapes") {
    val m12 = nn.ConvTranspose2d(
      in_channels = 16,
      out_channels = 33,
      kernel_size = (3, 5),
      stride = (2, 1),
      padding = (4, 2)
    ) // (5, 7), padding_mode = "reflect")
    val input = torch.randn(Seq(20, 16, 50, 100)) // , 64, 8, 9))
    assertEquals(m12(input).shape, Seq(20, 33, 93, 100)) // , 64, 5, 7))

  }
}

class ConvTranspose2dSuite3 extends munit.FunSuite {
  test("ConvTranspose2d output shapes") {
    val m12 = nn.ConvTranspose2d(
      in_channels = 16,
      out_channels = 33,
      kernel_size = (3, 5),
      stride = (2, 1),
      padding = (4, 2)
    ) // (5, 7), padding_mode = "reflect")
    val input = torch.randn(Seq(1, 16, 12, 12)) // , 64, 8, 9))
    val downsample = nn.Conv2d(16, 16, (3, 3), stride = 2, padding = 1)
    val unsample = nn.ConvTranspose2d(16, 16, 3, stride = 2, padding = 1)
    val h = downsample(input)
    println(s"h size ${h.shape}")
    assertEquals(downsample(input).shape, Seq(1, 16, 6, 6)) // , 33, 93, 100))
    val size = input.size
    val output = unsample(input = h, output_size = size)
    println(s"output shape ${output.shape}")
    //    val output = unsample(h,output_size = input.size())
    assertEquals(output.shape, Seq(1, 16, 12, 12)) // , 64, 5, 7))

  }
}

class ConvTranspose3dSuite extends munit.FunSuite {
  test("ConvTranspose3d output shapes") {
    val m12 = nn.ConvTranspose3d(
      in_channels = 16,
      out_channels = 33,
      kernel_size = (3, 5, 2),
      stride = (2, 1, 1),
      padding = (0, 4, 2)
    ) // (5, 7), padding_mode = "reflect")
    val input = torch.randn(Seq(20, 16, 10, 50, 100)) // , 16, 12, 12)) //, 64, 8, 9))
    //    val downsample = nn.Conv2d(16, 16, 3, stride = 2, padding = 1)
    //    val unsample = nn.ConvTranspose2d(16, 16, 3, stride = 2, padding = 1)
    //    val h = downsample(input)
    //    println(s"h size ${h.shape}")
    //    assertEquals(downsample(input).shape, Seq(1, 16, 6, 6)) //, 33, 93, 100))
    val size = input.size
    val output = m12(input) // unsample(input = h, output_size = size)
    println(s"output shape ${output.shape}")
    //    val output = unsample(h,output_size = input.size())
    assertEquals(output.shape, Seq(20, 33, 21, 46, 97)) // , 12, 12)) //, 64, 5, 7))

  }
}

class ConvolutionSuite extends munit.FunSuite {

  test("mismatchShapeConv2d") {
    val dtypes = List[FloatNN | ComplexNN](torch.float32, torch.complex64)
    for (dtype <- dtypes) {
      val x = torch.randn(Seq(1, 10, 1, 28, 28), dtype)
      val w = torch.randn(Seq(6, 1, 5, 5), dtype)

      intercept[RuntimeException](F.conv2d(x, w))
      // TODO find a way to run interceptMessage comparing only the first line/string prefix as we don't care about the c++ stacktrace here
      //   interceptMessage[RuntimeException] {
      //     """Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [1, 10, 1, 28, 28]"""
      //   } {
      //     conv2d(x, w)
      //   }
    }
  }

}
