package torch.nn.modules

import org.bytedeco.javacpp.{BoolPointer, DoublePointer, FloatPointer, LongPointer}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch.ScalarType
import org.bytedeco.pytorch.*
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn
import torch.nn.modules.normalization.LocalResponseNorm
import torch.nn.modules.regularization.Upsample.UpsampleMode
import torch.nn.modules.sparse.EmbeddingBag.EmbeddingBagMode

object otherSuite

import org.bytedeco.javacpp.{BoolPointer, DoublePointer, LongPointer}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  CosineSimilarityImpl,
  CosineSimilarityOptions,
  ModuleListImpl,
  PairwiseDistanceImpl,
  PairwiseDistanceOptions,
  SequentialImpl,
  TransformerImpl
}
import torch.internal.NativeConverters.{fromNative, toNative}

class CosineSimilarityRawSuite extends munit.FunSuite {
  test("CosineSimilarityImpl output shapes") {
    val input1 = torch.randn(Seq(100, 128)).native
    val input2 = torch.randn(Seq(100, 128)).native
    val options = new CosineSimilarityOptions()
    options.dim().put(LongPointer(1).put(1))
    options.eps().put(DoublePointer(1).put(1e-6))
    val model = CosineSimilarityImpl(options)
    val output = model.forward(input1, input2)
    assertEquals(fromNative(output).shape, Seq(100))
  }
}

class CosineSimilaritySuite extends munit.FunSuite {
  test("CosineSimilaritySuite output shapes") {
    val m12 = nn.CosineSimilarity(dim = 1, eps = 1e-6)
    val input1 = torch.randn(Seq(100, 128))
    val input2 = torch.randn(Seq(100, 128))
    assertEquals(m12(input1, input2).shape, Seq(100)) // torch.Size([100]) //错误得到128
    //    println(m12(input))
  }
}

class CosineSimilaritySuites extends munit.FunSuite {
  test("CosineSimilaritySuites output shapes") {
    val m12 = nn.CosineSimilarity(dim = 1, eps = 1e-6)
    val input1 = torch.randn(Seq(100, 128))
    val input2 = torch.randn(Seq(100, 128))
    assertEquals(m12(input1, input2).shape, Seq(100)) // torch.Size([100]) //错误得到128
    //    println(m12(input))
  }
}

class PairwiseDistanceSuite extends munit.FunSuite {
  test("PairwiseDistanceSuite output shapes") {
    val m12 = nn.PairwiseDistance(p = 2)
    val input1 = torch.randn(Seq(100, 128))
    val input2 = torch.randn(Seq(100, 128))
    assertEquals(m12(input1, input2).shape, Seq(100)) // torch.Size([100])
    //    println(m12(input))
  }
}

//>>> input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
//>>> input
//tensor([[[[1., 2.],
//          [3., 4.]]]])
//
//>>> m = nn.Upsample(scale_factor=2, mode='nearest')
//>>> m(input)
class UpsampleRawSuite extends munit.FunSuite {
  test("UpsampleRawSuite output shapes") {
    System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
    val input = torch.arange(1, 5, dtype = torch.float32).view(1, 1, 2, 2)
    val input3 = torch.zeros(Seq(3, 3)).view(1, 1, 3, 3)
    val options: UpsampleOptions = UpsampleOptions()
    options.scale_factor().put(DoubleVector(1).put(2))
    options.mode().put(new kNearest())
    println(s"input: ${input}")
    val model = UpsampleImpl(options)

    val output = fromNative(model.forward(input.native.to(ScalarType.Float)))
    println(s"output ${output.shape}")

//    options.align_corners().put(false)
//    options.mode().put(new kBilinear())
//    options.size().put(LongPointer(3,3))
//    println(s"  align ${model.options().align_corners().get()}  scala factor ${model.options().scale_factor().get().get(0)} factor2  mode ${model.options().mode()}")
//    val m13 = nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners = Some(true))
//    val size = m13.nativeModule.options().size()
//    println(s"  align ${m13.nativeModule.options().align_corners().get()}  scala factor ${m13.nativeModule.options().scale_factor().get().get(0)} mode ${m13.nativeModule.options().mode()}")
//    //    println(s"size ${size.get()} ")
//    assertEquals(m13(input3).shape, Seq(1, 1, 4, 4))
//    println(m13(input))
  }
}

//java.lang.RuntimeException: unflatten: Provided sizes [0] don't multiply up to the size of dim 1 (50) in the input tensor
class UnflattenSuite extends munit.FunSuite {
  test("UnflattenSuite output shapes") {
    val inp = torch.randn(Seq(1, 3, 10, 12))
    val w = torch.randn(Seq(2, 3, 4, 5))
    val input = torch.randn(Seq(2, 50)) // ,names = ("N","features"))
    val m12 = nn.Unflatten(
      dim_name = Some("features"),
      named_shape = Some(Map("C" -> 2, "H" -> 5, "W" -> 5))
    )
    println(s"m12(input) ${m12(input).shape}")
    assertEquals(m12(input).shape, Seq(2, 2, 5, 5)) // Actual   :ArraySeq(2, 50)
  }
}

//align true scala factor
//2.0 mode org.bytedeco.pytorch.UpsampleMode[address = 0x13f2f39ed60
//java.lang.RuntimeException: invalid vector subscript
//at org bytedeco.pytorch.UpsampleImpl.forward(Native Method)
//java.lang.NullPointerException: Cannot invoke "org.bytedeco.javacpp.BytePointer.put(org.bytedeco.javacpp.Pointer)" because the return value of "org.bytedeco.pytorch.UnflattenOptions.dimname()" is null
class Upsample1Suite extends munit.FunSuite {
  test("Upsample1Suite output shapes") {
    System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
    val input = torch.arange(1, 5, dtype = torch.float32).view(1, 1, 2, 2)

    val input3 = torch.zeros(Seq(3, 3)).view(1, 1, 3, 3)
    //    val m12 =nn.Upsample(size =None,scale_factor =2 ,mode ="nearest")
    val m13 = nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners = Some(true))
    //    val  m14 = nn.Upsample(scale_factor = 2, mode = "bilinear") //,align_corners = Some(true))
    //    val input = torch.randn(Seq(20, 16, 4, 32, 32))
    //    println(s"upsample ${m13.nativeModule.options().mode().get0()}")
    val size = m13.nativeModule.options().size()

    println(
      s"  align ${m13.nativeModule.options().align_corners().get()}  scala factor ${m13.nativeModule
          .options()
          .scale_factor()
          .get()
          .get(0)} mode ${m13.nativeModule.options().mode()}"
    )
//    println(s"size ${size.get()} ")
    assertEquals(m13(input3).shape, Seq(1, 1, 4, 4))
    println(m13(input))
  }
}

class Unflatten2Suite extends munit.FunSuite {
  test("Unflatten2Suite output shapes") {
    //    val inp = torch.randn(Seq(1, 3, 10, 12))
    //    val w = torch.randn(Seq(2, 3, 4, 5))
    val input = torch.randn(Seq(2, 196)) // ,names = ("N","features"))
    val m12 = nn.Unflatten(dim = 1, unflattened_size = (14, 14))
    println(s"m12(input) ${m12(input).shape}")
    assertEquals(m12(input).shape, Seq(2, 14, 14))
  }
}
class FoldSuite extends munit.FunSuite {
  test("FoldSuite output shapes") {
    val m12 =
      nn.Fold(output_size = (4, 5), kernel_size = (2, 2)) // (5, 7), padding_mode = "reflect")
    val input = torch.randn(Seq(1, 3 * 2 * 2, 12))
    println(s"m12(input) ${m12(input).shape}")
    assertEquals(m12(input).shape, Seq(1, 3, 4, 5))
  }
}

class UnfoldSuite extends munit.FunSuite {
  test("UnfoldSuite output shapes") {
    val m12 = nn.Unfold(kernel_size = (2, 3)) // (5, 7), padding_mode = "reflect")
    val input = torch.randn(Seq(2, 5, 3, 4))
    val output = m12(input)
    println(s"m12(input) ${output.shape}")
    assertEquals(m12(input).shape, Seq(2, 30, 4))
  }
}
