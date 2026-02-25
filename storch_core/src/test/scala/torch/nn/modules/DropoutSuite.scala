package torch.nn.modules

import torch.nn

class DropoutSuite extends munit.FunSuite {
  test("DropoutSuite output shapes") {
    val m12 = nn.Dropout(p = 0.2)
    val input = torch.randn(Seq(20, 16))
    assertEquals(m12(input).shape, Seq(20, 16))
    println(m12(input))
  }
}

class Dropout2dSuite extends munit.FunSuite {
  test("Dropout2dSuite output shapes") {
    val m12 = nn.Dropout2d(p = 0.2)
    val input = torch.randn(Seq(20, 16, 29))
    assertEquals(m12(input).shape, Seq(20, 16, 29))
    println(m12(input))
  }
}

class Dropout3dSuite extends munit.FunSuite {
  test("Dropout3dSuite output shapes") {
    val m12 = nn.Dropout3d(p = 0.2)
    val input = torch.randn(Seq(20, 16, 4, 32, 32))
    assertEquals(m12(input).shape, Seq(20, 16, 4, 32, 32))
    println(m12(input))
  }
}

class AlphaDropoutSuite extends munit.FunSuite {
  test("AlphaDropoutSuite output shapes") {
    val m12 = nn.AlphaDropout(p = 0.2)
    val input = torch.randn(Seq(20, 16))
    assertEquals(m12(input).shape, Seq(20, 16))
    println(m12(input))
  }
}

class FeatureAlphaDropoutSuite extends munit.FunSuite {
  test("FeatureAlphaDropout output shapes") {
    val m12 = nn.FeatureAlphaDropout(p = 0.2)
    val input = torch.randn(Seq(20, 16, 4, 32, 32))
    assertEquals(m12(input).shape, Seq(20, 16, 4, 32, 32))
    println(m12(input))
  }
}
