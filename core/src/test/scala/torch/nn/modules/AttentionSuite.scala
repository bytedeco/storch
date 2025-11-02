package torch.nn.modules

import org.bytedeco.javacpp.{BoolPointer, DoublePointer, LongPointer}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.*
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn

class AttentionSuite

//https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer
//
// scala.NotImplementedError: an implementation is missing
class transformerSuite extends munit.FunSuite {
  test("transformerSuite output shapes") {
    val transformer = nn.Transformer(nhead = 16, num_encoder_layers = 12)
    val src = torch.rand(Seq(10, 32, 512))
    val tgt = torch.rand(Seq(20, 32, 512))
    val out = transformer(src, tgt)
    println(s"transformer out shape ${out.shape}")
    val input = torch.randn(Seq(1, 64, 8, 9)) // transformer out shape ArraySeq(20, 32, 512)
  }
}

//java.lang.RuntimeException: Passing an empty index list to Tensor::index() is not valid syntax
//multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
//attn_output
//, attn_output_weights = multihead_attn(query, key, value)
class multiHeadCoderSuite extends munit.FunSuite {
  test("multiHeadCoderSuite output shapes") {
    //    val m1 = nn.TransformerEncoderLayer(d_model = 8,n_head = 4)
    val batchSize = 2
    val seqLength = 10
    val embedDim = 64
    val numHeads = 8
    val multiheadAttention = nn.MultiheadAttention(
      embed_dim = 64,
      num_heads = 8,
      dropout = 0.1f
    ) // ,kdim = 8,bias = true,vdim = 8) //java.lang.RuntimeException: from is out of bounds for float
    val input = torch.randn(Seq(batchSize, seqLength, embedDim))
    val out = multiheadAttention(input, input, input)
    println(s"multiheadAttention attn_output ${out._1.shape} attn_weight ${out._2.shape}")
  }
}

class TransformerEncoderLayerSuite extends munit.FunSuite {
  test("TransformerEncoderLayer output shapes") {
    val input = torch.randn(Seq(10, 32, 512))
    //    assertEquals(m12(input).shape, Seq(1, 64, 5, 7)) //lack of  paramter
    val layer = nn.TransformerEncoderLayer(
      d_model = 512,
      n_head = 8,
      dim_feedforward = 2048,
      dropout = 0.1f,
      activation = "relu"
    )
    //    val encoder = nn.TransformerEncoder(encoder_layer = layer, num_layers = 6)
    val out = layer(input)
    println(s"out .shape ${out.shape}")

  }
}

class TransformerEncoderLayerSuite2 extends munit.FunSuite {
  test("TransformerEncoderLayer output shapes") {
    val input = torch.randn(Seq(32, 10, 512))
    //    assertEquals(m12(input).shape, Seq(1, 64, 5, 7)) //lack of  paramter
    val layer = nn.TransformerEncoderLayer(d_model = 512, n_head = 8, batch_first = true)
    //    val encoder = nn.TransformerEncoder(encoder_layer = layer, num_layers = 6)
    val out = layer(input)
    println(s"out .shape ${out.shape}")

  }
}

//https://pytorch.org/docs/2.2/generated/torch.nn.MultiheadAttention.html
class TransformerDecoderLayerSuite extends munit.FunSuite {
  test("TransformerDecoderLayerSuite output shapes") {

    val layer = nn.TransformerDecoderLayer(d_model = 512, n_head = 8)
    //    val decoder = nn.TransformerDecoder(decoder_layer = layer, num_layers = 6)
    val memory = torch.randn(Seq(10, 32, 512))
    val tgt = torch.randn(Seq(20, 32, 512))
    val out = layer(tgt, memory)
    println(s"out.shape ${out.shape}")

  }
}

//https://pytorch.org/docs/2.2/generated/torch.nn.MultiheadAttention.html
class TransformerDecoderLayerSuite2 extends munit.FunSuite {
  test("TransformerDecoderLayerSuite2 output shapes") {

    val layer = nn.TransformerDecoderLayer(d_model = 512, n_head = 8, batch_first = true)
    //    val decoder = nn.TransformerDecoder(decoder_layer = layer, num_layers = 6)
    val memory = torch.randn(Seq(32, 10, 512))
    val tgt = torch.randn(Seq(32, 20, 512))
    val out = layer(tgt, memory)
    println(s"out.shape ${out.shape}")

  }
}
class TransformerEncoderSuite extends munit.FunSuite {
  test("TransformerEncoderLayer output shapes") {
    val input = torch.randn(Seq(10, 32, 512))
    //    assertEquals(m12(input).shape, Seq(1, 64, 5, 7)) //lack of  paramter
    val layer = nn.TransformerEncoderLayer(d_model = 512, n_head = 8)
    val encoder = nn.TransformerEncoder(encoder_layer = layer, num_layers = 6)
    val out = encoder(input)
    println(s"out .shape ${out.shape}")

  }
}

//https://pytorch.org/docs/2.2/generated/torch.nn.MultiheadAttention.html
class TransformerDecoderSuite extends munit.FunSuite {
  test("TransformerDecoderSuite output shapes") {

    val layer = nn.TransformerDecoderLayer(d_model = 512, n_head = 8)
    val decoder = nn.TransformerDecoder(decoder_layer = layer, num_layers = 6)
    val memory = torch.randn(Seq(10, 32, 512))
    val tgt = torch.randn(Seq(20, 32, 512))
    val out = decoder(tgt, memory)
    println(s"out.shape ${out.shape}")
    assertEquals(out.shape, Seq(20, 32, 512))

  }
}

class MultiHeadRawSuite extends munit.FunSuite {
  test("MultiHeadRawSuite output shapes") {

    //    val m1 = nn.TransformerEncoderLayer(d_model = 8,n_head = 4)
    // Expected input len1 == embed_dim && len2 == key.size(-1) to be true but got false.
    val batchSize = 8
    val seqLength = 8
    val embedDim = 8
    val numHeads = 8
    val dropout = 0.1f

    val kDim: Int | Option[Int] = None // Some(8)
    val vDim: Int | Option[Int] = None // Some(8)
    val bias: Boolean = true
    val addBiasKV: Boolean = false
    val addZeroAttn: Boolean = false
    val batchFirst: Boolean = false
    val input = torch.randn(Seq(batchSize, seqLength, embedDim))
    val options = new MultiheadAttentionOptions(embedDim.toLong, numHeads.toLong)
    options.embed_dim().put(LongPointer(1).put(embedDim.toLong))
    options.num_heads().put(LongPointer(1).put(numHeads.toLong))
    options.dropout().put(DoublePointer(1).put(dropout.toDouble))
    options.bias().put(bias)
    options.add_bias_kv().put(addBiasKV)
    options.add_zero_attn().put(addZeroAttn)
    kDim match {
      case k: Int => options.kdim().put(k.toLong)
      case k: Option[Int] =>
        if k.isDefined then options.kdim().put(LongPointer(1).put(k.get.toLong))
        else options.kdim().put(LongPointer(1).put(embedDim.toLong))
    }
    vDim match {
      case v: Int => options.vdim().put(v.toLong)
      case v: Option[Int] =>
        if v.isDefined then options.vdim().put(LongPointer(1).put(v.get.toLong))
        else options.vdim().put(LongPointer(1).put(embedDim.toLong))
    }

    //    options.kdim().put(LongPointer(1).put(kDim.toLong))
    //    options.vdim().put(LongPointer(1).put(vDim.toLong))
    val model = MultiheadAttentionImpl(options)
    val output = model.forward(input.native, input.native, input.native)
    println(
      s"multiheadAttention attn_output ${fromNative(output.get0()).shape} attn_weight ${fromNative(output.get1()).shape}"
    )
    //    val multiheadAttention = nn.MultiheadAttention(embed_dim = 8,num_heads = 8,dropout = 0.1,kdim = 8,bias = true,vdim = 8) //java.lang.RuntimeException: from is out of bounds for float
    ////          val input = torch.randn(Seq(batchSize,seqLength,embedDim))
    //    val out = multiheadAttention(input,input,input)
    //    println(s"multiheadAttention attn_output ${out._1.shape} attn_weight ${out._2.shape}")

  }
}

class transformerEnCoderLayerRawSuite extends munit.FunSuite {
  test("transformerEnCoderLayerRawSuite output shapes") {
    //    System.setProperty("org.bytedeco.javacpp.logger.debug", "true")
    val dimFeedforward: Int = 2048
    val dropout: Double = 0.1
    //    val activation: TransformerActivation = TransformerActivation.kReLU
    val layerNormEps: Double = 1e-5
    val batchFirst: Boolean = false
    val normFirst: Boolean = false
    val bias: Boolean = true
    val dModel: Int = 512
    val nHead: Int = 8
    val numLayers = 4
    //    val input = torch.randn(Seq(32,10, 512))
    val input = torch.randn(Seq(10, 32, 512))
    //    assertEquals(m12(input).shape, Seq(1, 64, 5, 7)) //lack of  paramter
    val options = new TransformerEncoderLayerOptions(dModel.toLong, nHead.toLong)
    options.d_model().put(dModel)
    options.nhead().put(nHead.toLong)
    options.dim_feedforward().put(LongPointer(1).put(dimFeedforward.toLong))
    options.dropout().put(DoublePointer(1).put(dropout))
    options.activation().put(new kReLU)
    val nativeModule: TransformerEncoderLayerImpl = TransformerEncoderLayerImpl(options)
    val out = fromNative(nativeModule.forward(input.native))
    val encoderOptions = TransformerEncoderOptions(options, 4)
    encoderOptions.num_layers().put(LongPointer(1).put(numLayers.toLong))

    val encoderModel = TransformerEncoderImpl(encoderOptions)
    val output = fromNative(encoderModel.forward(input.native))
    println(s"out .shape ${out.shape} output shape ${output.shape}")

  }
}

//https://pytorch.org/docs/2.3/generated/torch.nn.TransformerDecoderLayer.html
//https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
//https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
//https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html
// java:suite://Tests
class transformerCoderSuite extends munit.FunSuite {
  test("transformerCoderSuite output shapes") {
    val m1 = nn.TransformerEncoderLayer(d_model = 8, n_head = 4)
    val m12 = nn.TransformerEncoder(encoder_layer = m1, num_layers = 4)
    val input = torch.randn(Seq(1, 64, 8, 9))
    val output = m12(input)
    assertEquals(m12(input).shape, Seq(1, 64, 5, 7)) // lack of  paramter
    val m2 = nn.TransformerDecoderLayer(d_model = 8, n_head = 4)
    val m22 = nn.TransformerDecoder(decoder_layer = m2, num_layers = 4)
    assertEquals(m22(output).shape, Seq(1, 64, 1, 1))
  }
}

//https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer
//
// scala.NotImplementedError: an implementation is missing
class transformerOldSuite extends munit.FunSuite {
  test("transformerOldSuite output shapes") {
    val transformer = nn.Transformer(nhead = 16, num_encoder_layers = 12, activation = "gelu")
    val src = torch.rand(Seq(10, 32, 512))
    val tgt = torch.rand(Seq(20, 32, 512))
    val out = transformer(src, tgt)
    println(s"transformer out shape ${out.shape}")
    val input = torch.randn(Seq(1, 64, 8, 9))
  }
}

//java.lang.RuntimeException: Passing an empty index list to Tensor::index() is not valid syntax
//multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
//attn_output
//, attn_output_weights = multihead_attn(query, key, value)
class multiHeadCoderOldSuite extends munit.FunSuite {
  test("multiHeadCoderSuite output shapes") {
    //    val m1 = nn.TransformerEncoderLayer(d_model = 8,n_head = 4)
    val batchSize = 2
    val seqLength = 10
    val embedDim = 64
    val numHeads = 8
    val multiheadAttention = nn.MultiheadAttention(
      embed_dim = 64,
      num_heads = 8,
      dropout = 0.1,
      kdim = 8,
      bias = true,
      vdim = 8
    ) // java.lang.RuntimeException: from is out of bounds for float
    val input = torch.randn(Seq(batchSize, seqLength, embedDim))
    val out = multiheadAttention(input, input, input)
    println(s"multiheadAttention attn_output ${out._1.shape} attn_weight ${out._2.shape}")
  }
}

class TransformerEncoderLayerOldSuite extends munit.FunSuite {
  test("TransformerEncoderLayer output shapes") {
    val input = torch.randn(Seq(10, 32, 512))
    //    assertEquals(m12(input).shape, Seq(1, 64, 5, 7)) //lack of  paramter
    val layer = nn.TransformerEncoderLayer(d_model = 512, n_head = 8)
    //    val encoder = nn.TransformerEncoder(encoder_layer = layer, num_layers = 6)
    val out = layer(input)
    println(s"out .shape ${out.shape}")

  }
}

class TransformerEncoderLayerOldSuite2 extends munit.FunSuite {
  test("TransformerEncoderLayer output shapes") {
    val input = torch.randn(Seq(32, 10, 512))
    //    assertEquals(m12(input).shape, Seq(1, 64, 5, 7)) //lack of  paramter
    val layer = nn.TransformerEncoderLayer(d_model = 512, n_head = 8, batch_first = true)
    //    val encoder = nn.TransformerEncoder(encoder_layer = layer, num_layers = 6)
    val out = layer(input)
    println(s"out .shape ${out.shape}")

  }
}

//https://pytorch.org/docs/2.2/generated/torch.nn.MultiheadAttention.html
class TransformerDecoderLayerOldSuite extends munit.FunSuite {
  test("TransformerDecoderLayerOldSuite output shapes") {

    val layer = nn.TransformerDecoderLayer(d_model = 512, n_head = 8)
    //    val decoder = nn.TransformerDecoder(decoder_layer = layer, num_layers = 6)
    val memory = torch.randn(Seq(10, 32, 512))
    val tgt = torch.randn(Seq(20, 32, 512))
    val out = layer(tgt, memory)
    println(s"out.shape ${out.shape}")

  }
}

//https://pytorch.org/docs/2.2/generated/torch.nn.MultiheadAttention.html
class TransformerDecoderLayerOldSuite2 extends munit.FunSuite {
  test("TransformerDecoderLayerOldSuite2 output shapes") {

    val layer = nn.TransformerDecoderLayer(d_model = 512, n_head = 8, batch_first = true)
    //    val decoder = nn.TransformerDecoder(decoder_layer = layer, num_layers = 6)
    val memory = torch.randn(Seq(32, 10, 512))
    val tgt = torch.randn(Seq(32, 20, 512))
    val out = layer(tgt, memory)
    println(s"out.shape ${out.shape}")

  }
}
class TransformerEncoderOldSuite extends munit.FunSuite {
  test("TransformerEncoderOldSuite output shapes") {
    val input = torch.randn(Seq(10, 32, 512))
    //    assertEquals(m12(input).shape, Seq(1, 64, 5, 7)) //lack of  paramter
    val layer = nn.TransformerEncoderLayer(d_model = 512, n_head = 8)
    val encoder = nn.TransformerEncoder(encoder_layer = layer, num_layers = 6)
    val out = encoder(input)
    println(s"out .shape ${out.shape}")

  }
}

//https://pytorch.org/docs/2.2/generated/torch.nn.MultiheadAttention.html
class TransformerDecoderOldSuite extends munit.FunSuite {
  test("TransformerDecoderOldSuite output shapes") {

    val layer = nn.TransformerDecoderLayer(d_model = 512, n_head = 8)
    val decoder = nn.TransformerDecoder(decoder_layer = layer, num_layers = 6)
    val memory = torch.randn(Seq(10, 32, 512))
    val tgt = torch.randn(Seq(20, 32, 512))
    val out = decoder(tgt, memory)
    println(s"out.shape ${out.shape}")

  }
}

//https://pytorch.org/docs/2.3/generated/torch.nn.TransformerDecoderLayer.html
//https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
//https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
//https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html
// java:suite://Tests
class transformerCoderOldSuite extends munit.FunSuite {
  test("transformerCoderOldSuite output shapes") {
    val m1 = nn.TransformerEncoderLayer(d_model = 8, n_head = 4)
    val m12 = nn.TransformerEncoder(encoder_layer = m1, num_layers = 4)
    val input = torch.randn(Seq(1, 64, 8, 9))
    assertEquals(m12(input).shape, Seq(1, 64, 5, 7)) // lack of  paramter
    val m2 = nn.TransformerDecoderLayer(d_model = 8, n_head = 4)
    val m22 = nn.TransformerDecoder(decoder_layer = m2, num_layers = 4)
    assertEquals(m22(input).shape, Seq(1, 64, 1, 1))
  }
}
