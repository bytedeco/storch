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
import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch.ScalarType
import org.bytedeco.pytorch.*
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn.modules.normalization.LocalResponseNorm
import torch.nn.modules.regularization.Upsample.UpsampleMode
import torch.nn.modules.sparse.EmbeddingBag.EmbeddingBagMode
import torch.{Tensor, nn, noGrad}

//>>> embedding = nn.Embedding(10, 3)
//>>> # a batch of 2 samples of 4 indices each
//>>> input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
//>>> embedding(input)
class EmbeddingSuite1 extends munit.FunSuite {
  test("EmbeddingSuite1 output shapes") {
    val m12 = nn.Embedding(10, 3)
    val input1 = torch.arange(1, 9).view(2, 4) // .to(torch.float32)
    val out = m12(input1.to(torch.int32))
    assertEquals(out.shape, Seq(2, 4, 3))
  }
}

//>>> embedding = nn.Embedding(10, 3, padding_idx=0)
//>>> input = torch.LongTensor([[0, 2, 0, 5]])
//>>> embedding(input)
class EmbeddingSuite2 extends munit.FunSuite {
  test("EmbeddingSuite1 output shapes") {
    val m12 = nn.Embedding(10, 3, padding_idx = Some(0))
    val input1 = torch.Tensor(Seq(Seq(0, 2, 0, 5))).to(torch.float32)
    val out = m12(input1)
    assertEquals(out.shape, Seq(1, 4, 3))
    println(m12.weight)
  }
}

import org.bytedeco.pytorch.global.torch as tch

class EmbeddingBagRawSuite extends munit.FunSuite {
  test("EmbeddingBagRawSuite output shapes") {
    val options = new EmbeddingBagOptions(10, 3)
    options.mode().put(new kSum)
    options.embedding_dim().put(3)
    options.num_embeddings().put(10)
    options.norm_type().put(2.0)
    options.include_last_offset().put(false)
    options.scale_grad_by_freq().put(false)
    options.sparse().put(false)
    //    options.max_norm().put(2d)
    //    options.padding_idx().put()
    //    val m123 = nn.EmbeddingBag(num_embeddings = val options = new EmbeddingBagOptions(numEmbeddings.toLong, embeddingDim.toLong)10, embedding_dim = 3, mode = "sum") // (5, 7), padding_mode = "reflect")

    val input23 = torch.Tensor(Seq(1, 2, 4, 5, 4, 3, 2, 9)).to(torch.int32) // 16, 10)) // indices
    val offsets = torch.Tensor(Seq(0.0, 4.0)).to(torch.int8) // weight
    val pre = torch.zeros(input23.size).to(torch.int32) // per_sample_weigh 16 short
    //    val offset = tch.new
    val model = EmbeddingBagImpl(options)
    println(s"input type ${input23.native.dtype().name().getString()} offsets ${offsets.native
        .dtype()
        .name()
        .getString()} pre ${pre.native.dtype().name().getString()}")
    val output = model.forward(
      input23.native.to(ScalarType.Long),
      offsets.native.to(ScalarType.Long),
      pre.native.to(ScalarType.Float)
    )
    println(s"output ${fromNative(output).shape}")
    //
    //    val size = torch.empty(input23.size)
    //    val output = m123(input23.to(m123.paramType), offsets.to(torch.int64), size.to(m123.paramType))
    //    println(s"m12(input) ${output.shape}")

    //    val model = EmbeddingBagOptions(10, 2).max_norm(2).norm_type(2.5).scale_grad_by_freq(true).sparse(true).mode(torch.kSum).padding_idx(1))
    //    torch.empty()
    //    assertEquals(m12(input,offsets).shape, Seq(2,3)) //, 64, 5, 7))
  }
}

class EmbeddingBagSuite extends munit.FunSuite {
  test("EmbeddingBagSuite output shapes") {
    val m123 = nn.EmbeddingBag(
      num_embeddings = 10,
      embedding_dim = 3,
      mode = "sum"
    ) // (5, 7), padding_mode = "reflect")
    val input23 =
      torch.Tensor(Seq(1, 2, 4, 5, 4, 3, 2, 9)).to(torch.float32) // 16, 10)) //, 64, 8, 9))
    val offsets = torch.Tensor(Seq(0, 4))
    val size = torch.empty(input23.size)
    val output = m123(input23.to(m123.paramType), offsets.to(torch.int64), size.to(m123.paramType))
    println(s"m12(input) ${output.shape}")

    //    val model = EmbeddingBagOptions(10, 2).max_norm(2).norm_type(2.5).scale_grad_by_freq(true).sparse(true).mode(torch.kSum).padding_idx(1))
    //    torch.empty()
    //    assertEquals(m12(input,offsets).shape, Seq(2,3)) //, 64, 5, 7))
  }
}

class EmbeddingSuite extends munit.FunSuite {

  test("Embedding") {
    {
      torch.manualSeed(0)
      val embedding = nn.Embedding(10, 3)
      // a batch of 2 samples of 4 indices each
      val input = torch.Tensor(Seq(Seq(1L, 2, 4, 5), Seq(4L, 3, 2, 9)))
      val output = embedding(input.to(embedding.paramType))
      val expectedOutput = Tensor(
        Seq(
          Seq(
            Seq(-0.4339f, 0.8487f, 0.6920f),
            Seq(-0.3160f, -2.1152f, 0.3223f),
            Seq(0.1198f, 1.2377f, -0.1435f),
            Seq(-0.1116f, -0.6136f, 0.0316f)
          ),
          Seq(
            Seq(0.1198f, 1.2377f, -0.1435f),
            Seq(-1.2633f, 0.3500f, 0.3081f),
            Seq(-0.3160f, -2.1152f, 0.3223f),
            Seq(0.0525f, 0.5229f, 2.3022f)
          )
        )
      )
      assert(torch.allclose(output, expectedOutput, atol = 1e-4))
    }
    {
      torch.manualSeed(0)
      // example with padding_idx
      val embedding = nn.Embedding(5, 3, padding_idx = Some(0))
      embedding.weight = Tensor(
        Seq(
          Seq(0f, 0f, 0f),
          Seq(0.5684f, -1.0845f, -1.3986f),
          Seq(0.4033f, 0.8380f, -0.7193f),
          Seq(0.4033f, 0.8380f, -0.7193f),
          Seq(-0.8567f, 1.1006f, -1.0712f)
        )
      )
      val input = torch.Tensor(Seq(Seq(0L, 2, 0, 4)))
      val output = embedding(input.to(embedding.paramType))

      val expectedOutput = Tensor(
        Seq(
          Seq(0f, 0f, 0f),
          Seq(0.4033f, 0.8380f, -0.7193f),
          Seq(0f, 0f, 0f),
          Seq(-0.8567f, 1.1006f, -1.0712f)
        )
      ).unsqueeze(0)
      assert(torch.allclose(output, expectedOutput, atol = 1e-4))
    }
    {
      torch.manualSeed(0)
      //  example of changing `pad` vector
      val paddingIdx = 0
      val embedding = nn.Embedding(3, 3, padding_idx = Some(paddingIdx))
      noGrad {
        embedding.weight(Seq(paddingIdx)) = torch.ones(3)
      }
      val expectedOutput = Tensor(
        Seq(
          Seq(1f, 1f, 1f),
          Seq(0.5684f, -1.0845f, -1.3986f),
          Seq(0.4033f, 0.8380f, -0.7193f)
        )
      )
      assert(torch.allclose(embedding.weight, expectedOutput, atol = 1e-4))
    }
  }

  test("EmbeddingBag") {
    {
      torch.manualSeed(0)
      val embedding = nn.EmbeddingBag(10, 3)
      // a batch of 2 samples of 4 indices each
      val input = torch.Tensor(Seq(Seq(1L, 2, 4, 5), Seq(4L, 3, 2, 9)))
      val output = embedding(input.to(embedding.paramType))
      val expectedOutput = Tensor(
        Seq(
          Seq(
            Seq(-0.4339f, 0.8487f, 0.6920f),
            Seq(-0.3160f, -2.1152f, 0.3223f),
            Seq(0.1198f, 1.2377f, -0.1435f),
            Seq(-0.1116f, -0.6136f, 0.0316f)
          ),
          Seq(
            Seq(0.1198f, 1.2377f, -0.1435f),
            Seq(-1.2633f, 0.3500f, 0.3081f),
            Seq(-0.3160f, -2.1152f, 0.3223f),
            Seq(0.0525f, 0.5229f, 2.3022f)
          )
        )
      )
      assert(torch.allclose(output, expectedOutput, atol = 1e-4))
    }
    {
      torch.manualSeed(0)
      // example with padding_idx
      val embedding = nn.Embedding(5, 3, padding_idx = Some(0))
      embedding.weight = Tensor(
        Seq(
          Seq(0f, 0f, 0f),
          Seq(0.5684f, -1.0845f, -1.3986f),
          Seq(0.4033f, 0.8380f, -0.7193f),
          Seq(0.4033f, 0.8380f, -0.7193f),
          Seq(-0.8567f, 1.1006f, -1.0712f)
        )
      )
      val input = torch.Tensor(Seq(Seq(0L, 2, 0, 4)))
      val output = embedding(input.to(embedding.paramType))

      val expectedOutput = Tensor(
        Seq(
          Seq(0f, 0f, 0f),
          Seq(0.4033f, 0.8380f, -0.7193f),
          Seq(0f, 0f, 0f),
          Seq(-0.8567f, 1.1006f, -1.0712f)
        )
      ).unsqueeze(0)
      assert(torch.allclose(output, expectedOutput, atol = 1e-4))
    }
    {
      torch.manualSeed(0)
      //  example of changing `pad` vector
      val paddingIdx = 0
      val embedding = nn.EmbeddingBag(3, 3, padding_idx = Some(paddingIdx))
      noGrad {
        embedding.weight(Seq(paddingIdx)) = torch.ones(3)
      }
      val expectedOutput = Tensor(
        Seq(
          Seq(1f, 1f, 1f),
          Seq(0.5684f, -1.0845f, -1.3986f),
          Seq(0.4033f, 0.8380f, -0.7193f)
        )
      )
      assert(torch.allclose(embedding.weight, expectedOutput, atol = 1e-4))
    }
  }
}
