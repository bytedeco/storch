package torch
package nn
package functional

import org.bytedeco.pytorch.{
  MultiheadAttentionForwardFuncOptions,
  TensorVector,
  TensorOptional,
  DoubleOptional
}
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.{fromNative, toArrayRef}
//import org.bytedeco.pytorch.ScalarTypeOptional
//import org.bytedeco.javacpp.{Pointer, CLongPointer}
//import org.bytedeco.javacpp.annotation.{Const, ByRef, ByVal, Namespace}
//import torch.internal.NativeConverters.*

//case class TensorTriple[T <: FloatNN | ComplexNN](
private[torch] trait Recurrent {

  def gru_cell[T, TT <: FloatNN | ComplexNN](
      input: Tensor[TT],
      hx: Tensor[TT],
      w_ih: Tensor[TT],
      w_hh: Tensor[TT],
      b_ih: Tensor[TT],
      b_hh: Tensor[TT]
  ): Tensor[TT] = {

    val inputNative = input.native
    val hxNative = hx.native // TensorVector(hx.map(_.native) *)
    val w_ihNative = w_ih.native
    val w_hhNative = w_hh.native
    val b_ihNative = TensorOptional(b_ih.native)
    val b_hhNative = TensorOptional(b_hh.native)
    //    val numLayersNative = CLongPointer( numLayers.toArray:_*)
    val native =
      torchNative.gru_cell(inputNative, hxNative, w_ihNative, w_hhNative, b_ihNative, b_hhNative)
    //    TensorTuple2(output = fromNative[TT](native.get0()), hx = fromNative(native.get1))

    //    val tensorTuple = TensorTuple2(fromNative[TT](native.get0()), fromNative[TT](native.get1()))
    fromNative(native)
    //    fromNative(torchNative.lstm(inputNative, hxNative, paramsNtive, has_biases, num_layers, dropout, train, bidirectional, batch_first))

  }

  //  inline Tensor torch::nn::utils::rnn:
  //  :invert_permutation(const Tensor &permutation)
  def invert_permutation[D <: DType](permutation: Tensor[D]): Tensor[D] = {

    fromNative(torchNative.invert_permutation(permutation.native))
  }

  def pad_sequence[D1 <: DType](tensorArray: Seq[Tensor[D1]]): Tensor[D1] =
    val tensorVector = TensorVector(tensorArray.map(_.native).toArray*)
    fromNative(torchNative.pad_sequence(tensorVector))

  def pad_sequence[T, TT <: FloatNN | ComplexNN](
      sequences: Seq[Tensor[TT]],
      batch_first: Boolean,
      padding_value: Double = 0,
      padding_side: String = "right"
  ): Tensor[TT] = {
    val sequencesNative = toArrayRef(sequences)
    val native = torchNative.pad_sequence(sequencesNative, batch_first, padding_value, padding_side)
    fromNative(native)
  }

  //
  def gru[T, TT <: FloatNN | ComplexNN](
      input: Tensor[TT],
      hx: Tensor[TT],
      batch_sizes: Tensor[TT],
      params: Option[Seq[Tensor[TT]]],
      has_biases: Boolean,
      num_layers: Long, // CLongPointer,
      dropout: Double,
      train: Boolean,
      bidirectional: Boolean,
      batch_first: Boolean
  ): TensorTuple2[TT] = {

    val inputNative = input.native
    //    val hxNative = toArrayRef(hx)
    val paramsNtive = toArrayRef(params.get)
    //    val numLayersNative = CLongPointer( numLayers.toArray:_*)
    val native = torchNative.gru(
      inputNative,
      hx.native,
      paramsNtive,
      has_biases,
      num_layers,
      dropout,
      train,
      bidirectional,
      batch_first
    )
    //    TensorTuple(values = fromNative[TT](native.get0()), indices = fromNative(native.get1))

    val tensorTuple = TensorTuple2(fromNative[TT](native.get0()), fromNative[TT](native.get1()))
    tensorTuple
    //    fromNative(torchNative.lstm(inputNative, hxNative, paramsNtive, has_biases, num_layers, dropout, train, bidirectional, batch_first))

  }

  //
  def lstm_cell[T, TT <: FloatNN | ComplexNN](
      input: Tensor[TT],
      hx: Seq[Tensor[TT]],
      w_ih: Tensor[TT],
      w_hh: Tensor[TT],
      b_ih: Tensor[TT],
      b_hh: Tensor[TT]
  ): TensorTuple2[TT] = {

    val inputNative = input.native
    val hxNative = TensorVector(hx.map(_.native)*)
    val w_ihNative = w_ih.native
    val w_hhNative = w_hh.native
    val b_ihNative = TensorOptional(b_ih.native)
    val b_hhNative = TensorOptional(b_hh.native)
    //    val numLayersNative = CLongPointer( numLayers.toArray:_*)
    val native =
      torchNative.lstm_cell(inputNative, hxNative, w_ihNative, w_hhNative, b_ihNative, b_hhNative)
    TensorTuple2(output = fromNative[TT](native.get0()), hx = fromNative(native.get1))

    val tensorTuple = TensorTuple2(fromNative[TT](native.get0()), fromNative[TT](native.get1()))
    tensorTuple
    //    fromNative(torchNative.lstm(inputNative, hxNative, paramsNtive, has_biases, num_layers, dropout, train, bidirectional, batch_first))

  }

  def lstm_cell[T, TT <: FloatNN | ComplexNN](
      input: Tensor[TT],
      hx: Seq[Tensor[TT]],
      w_ih: Tensor[TT],
      w_hh: Tensor[TT]
  ): TensorTuple2[TT] = {

    val inputNative = input.native
    val hxNative = toArrayRef(hx)
    val w_ihNative = w_ih.native
    val w_hhNative = w_hh.native
    //    val numLayersNative = CLongPointer( numLayers.toArray:_*)
    val native = torchNative.lstm_cell(inputNative, hxNative, w_ihNative, w_hhNative)
    TensorTuple2(output = fromNative[TT](native.get0()), hx = fromNative(native.get1))

    val tensorTuple = TensorTuple2(fromNative[TT](native.get0()), fromNative[TT](native.get1()))
    tensorTuple
    //    fromNative(torchNative.lstm(inputNative, hxNative, paramsNtive, has_biases, num_layers, dropout, train, bidirectional, batch_first))

  }

  def lstm[T, TT <: FloatNN | ComplexNN](
      input: Tensor[TT],
      hx: Seq[Tensor[TT]],
      params: Option[Seq[Tensor[TT]]],
      has_biases: Boolean,
      num_layers: Long, // CLongPointer,
      dropout: Double,
      train: Boolean,
      bidirectional: Boolean,
      batch_first: Boolean
  ): TensorTriple[TT] = {

    val inputNative = input.native
    val hxNative = toArrayRef(hx)
    val paramsNtive = toArrayRef(params.get)
    //    val numLayersNative = CLongPointer( numLayers.toArray:_*)
    val native = torchNative.lstm(
      inputNative,
      hxNative,
      paramsNtive,
      has_biases,
      num_layers,
      dropout,
      train,
      bidirectional,
      batch_first
    )
    //    TensorTuple(values = fromNative[TT](native.get0()), indices = fromNative(native.get1))

    val tensorTriple = TensorTriple(
      fromNative[TT](native.get0()),
      fromNative[TT](native.get1()),
      fromNative[TT](native.get2())
    )
    tensorTriple
    //    fromNative(torchNative.lstm(inputNative, hxNative, paramsNtive, has_biases, num_layers, dropout, train, bidirectional, batch_first))

  }

  def lstm[T, TT <: FloatNN | ComplexNN](
      data: Tensor[TT],
      batch_size: Int,
      hx: Seq[Tensor[TT]],
      params: Option[Seq[Tensor[TT]]],
      has_biases: Boolean,
      num_layers: Long, // CLongPointer,
      dropout: Double,
      train: Boolean,
      bidirectional: Boolean,
      batch_first: Boolean
  ): TensorTriple[TT] = {

    val dataNative = data.native
    val hxNative = toArrayRef(hx)
    val batchSizeNative = torchNative.tensor(batch_size)
    val paramsNtive = toArrayRef(params.get)
    //    val numLayersNative = CLongPointer( numLayers.toArray:_*)
    val native = torchNative.lstm(
      dataNative,
      batchSizeNative,
      hxNative,
      paramsNtive,
      has_biases,
      num_layers,
      dropout,
      train,
      bidirectional
    )
    //  TensorTuple(values = fromNative[TT](native.get0()), indices = fromNative(native.get1))

    val tensorTriple = TensorTriple(
      fromNative[TT](native.get0()),
      fromNative[TT](native.get1()),
      fromNative[TT](native.get2())
    )
    tensorTriple
    //    fromNative(torchNative.lstm(inputNative, hxNative, paramsNtive, has_biases, num_layers, dropout, train, bidirectional, batch_first))

  }

  def scaled_dot_product_attention[D <: DType](
      query: Tensor[D],
      key: Tensor[D],
      value: Tensor[D],
      attn_mask: Option[Tensor[D]] = None,
      dropout_p: Double = 0.0,
      is_causal: Boolean = false,
      scale: Option[Double] = None,
      enable_gqa: Boolean = false
  ): Tensor[D] =
    scaledDotProductAttention(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

  //  scaled_dot_product_attention(query, key, value, attn_mask = None, dropout_p = 0.0,
  //    is_causal = False, scale = None, enable_gqa = False) -> Tensor:
  def scaledDotProductAttention[D <: DType](
      query: Tensor[D],
      key: Tensor[D],
      value: Tensor[D],
      attn_mask: Option[Tensor[D]] = None,
      dropout_p: Double = 0.0,
      is_causal: Boolean = false,
      scale: Option[Double] = None,
      enable_gqa: Boolean = false
  ): Tensor[D] = {
    val attnMaskOption =
      if attn_mask.isDefined then TensorOptional(attn_mask.get.native) else TensorOptional()
    val scaleOption =
      if scale.isDefined then new DoubleOptional(scale.get) else new DoubleOptional()
    fromNative(
      torchNative.scaled_dot_product_attention(
        query.native,
        key.native,
        value.native,
        attnMaskOption,
        dropout_p,
        is_causal,
        scaleOption,
        enable_gqa
      )
    )
  }

  def multi_head_attention_forward[T, D, TT <: FloatNN | ComplexNN](
      query: Tensor[TT],
      key: Tensor[TT],
      value: Tensor[TT],
      embed_dim_to_check: Long,
      num_heads: Long,
      in_proj_weight: Tensor[TT],
      in_proj_bias: Tensor[TT],
      bias_k: Tensor[TT],
      bias_v: Tensor[TT],
      add_zero_attn: Boolean,
      dropout_p: Double,
      out_proj_weight: Tensor[TT],
      out_proj_bias: Tensor[TT],
      training: Boolean,
      key_padding_mask: Tensor[TT],
      need_weights: Boolean,
      attn_mask: Tensor[TT],
      use_separate_proj_weight: Boolean,
      q_proj_weight: Tensor[TT],
      k_proj_weight: Tensor[TT],
      v_proj_weight: Tensor[TT],
      static_k: Tensor[TT],
      static_v: Tensor[TT],
      average_attn_weights: Boolean
  ): TensorTuple2[TT] = {

    val queryNative = query.native
    val keyNative = key.native
    val valueNative = value.native
    val options = MultiheadAttentionForwardFuncOptions(
      embed_dim_to_check,
      num_heads,
      in_proj_weight.native,
      in_proj_bias.native,
      bias_k.native,
      bias_v.native,
      add_zero_attn,
      dropout_p,
      out_proj_weight.native,
      out_proj_bias.native
    )

    options.training().put(training)
    options.key_padding_mask().put(key_padding_mask.native)
    options.need_weights().put(need_weights)
    options.attn_mask().put(attn_mask.native)
    options.use_separate_proj_weight().put(use_separate_proj_weight)
    options.average_attn_weights().put(average_attn_weights)
    options.static_k().put(static_k.native)
    options.static_v().put(static_v.native)
    options.q_proj_weight().put(q_proj_weight.native)
    options.k_proj_weight().put(k_proj_weight.native)
    options.v_proj_weight().put(v_proj_weight.native)

    //    val numLayersNative = CLongPointer( numLayers.toArray:_*)
    val native =
      torchNative.multi_head_attention_forward(queryNative, keyNative, valueNative, options)
    //    TensorTuple(values = fromNative[TT](native.get0()), indices = fromNative(native.get1))

    val tensorTuple2 = TensorTuple2(fromNative[TT](native.get0()), fromNative[TT](native.get1()))
    tensorTuple2
    //    fromNative(torchNative.lstm(inputNative, hxNative, paramsNtive, has_biases, num_layers, dropout, train, bidirectional, batch_first))

  }

  def rnn_tanh[T, TT <: FloatNN | ComplexNN](
      input: Tensor[TT],
      hx: Tensor[TT],
      params: Option[Seq[Tensor[TT]]],
      has_biases: Boolean,
      num_layers: Long,
      dropout: Double,
      train: Boolean,
      bidirectional: Boolean,
      batch_first: Boolean
  ): TensorTuple2[TT] = {

    val inputNative = input.native
    val hxNative = hx.native
    val paramsNtive = toArrayRef(params.get)
    val native = torchNative.rnn_tanh(
      inputNative,
      hxNative,
      paramsNtive,
      has_biases,
      num_layers,
      dropout,
      train,
      bidirectional,
      batch_first
    )
    TensorTuple2(output = fromNative[TT](native.get0()), hx = fromNative(native.get1))
    val tensorTuple = TensorTuple2(fromNative[TT](native.get0()), fromNative[TT](native.get1()))
    tensorTuple
  }

  def rnn_relu[T, TT <: FloatNN | ComplexNN](
      input: Tensor[TT],
      hx: Tensor[TT],
      params: Option[Seq[Tensor[TT]]],
      has_biases: Boolean,
      num_layers: Long,
      dropout: Double,
      train: Boolean,
      bidirectional: Boolean,
      batch_first: Boolean
  ): TensorTuple2[TT] = {

    val inputNative = input.native
    val hxNative = hx.native
    val paramsNtive = toArrayRef(params.get)
    val native = torchNative.rnn_relu(
      inputNative,
      hxNative,
      paramsNtive,
      has_biases,
      num_layers,
      dropout,
      train,
      bidirectional,
      batch_first
    )
    TensorTuple2(output = fromNative[TT](native.get0()), hx = fromNative(native.get1))
    val tensorTuple = TensorTuple2(fromNative[TT](native.get0()), fromNative[TT](native.get1()))
    tensorTuple
  }

  def rnn_tanh_cell[T, TT <: FloatNN | ComplexNN](
      input: Tensor[TT],
      hx: Tensor[TT],
      w_ih: Tensor[TT],
      w_hh: Tensor[TT]
  ): Tensor[TT] = {

    val inputNative = input.native
    val hxNative = hx.native
    val w_ihNative = w_ih.native
    val w_hhNative = w_hh.native
    val native = torchNative.rnn_tanh_cell(inputNative, hxNative, w_ihNative, w_hhNative)
    fromNative[TT](native)
  }

  def rnn_tanh_cell[T, TT <: FloatNN | ComplexNN](
      input: Tensor[TT],
      hx: Tensor[TT],
      w_ih: Tensor[TT],
      w_hh: Tensor[TT],
      b_ih: Option[Tensor[TT]],
      b_hh: Option[Tensor[TT]]
  ): Tensor[TT] = {

    val inputNative = input.native
    val hxNative = hx.native
    val w_ihNative = w_ih.native
    val w_hhNative = w_hh.native
    val b_ihNative =
      if b_ih.isDefined then new TensorOptional(b_ih.get.native) else new TensorOptional()
    val b_hhNative =
      if b_hh.isDefined then new TensorOptional(b_hh.get.native) else new TensorOptional()
    val native = torchNative.rnn_tanh_cell(
      inputNative,
      hxNative,
      w_ihNative,
      w_hhNative,
      b_ihNative,
      b_hhNative
    )
    fromNative[TT](native)
  }

  def rnn_relu_cell[T, TT <: FloatNN | ComplexNN](
      input: Tensor[TT],
      hx: Tensor[TT],
      w_ih: Tensor[TT],
      w_hh: Tensor[TT]
  ): Tensor[TT] = {

    val inputNative = input.native
    val hxNative = hx.native
    val w_ihNative = w_ih.native
    val w_hhNative = w_hh.native
    val native = torchNative.rnn_relu_cell(inputNative, hxNative, w_ihNative, w_hhNative)
    fromNative[TT](native)
  }

  def rnn_relu_cell[T, TT <: FloatNN | ComplexNN](
      input: Tensor[TT],
      hx: Tensor[TT],
      w_ih: Tensor[TT],
      w_hh: Tensor[TT],
      b_ih: Option[Tensor[TT]],
      b_hh: Option[Tensor[TT]]
  ): Tensor[TT] = {

    val inputNative = input.native
    val hxNative = hx.native
    val w_ihNative = w_ih.native
    val w_hhNative = w_hh.native
    val b_ihNative =
      if b_ih.isDefined then new TensorOptional(b_ih.get.native) else new TensorOptional()
    val b_hhNative =
      if b_hh.isDefined then new TensorOptional(b_hh.get.native) else new TensorOptional()
    val native = torchNative.rnn_relu_cell(
      inputNative,
      hxNative,
      w_ihNative,
      w_hhNative,
      b_ihNative,
      b_hhNative
    )
    fromNative[TT](native)
  }

}

case class TensorTuple2[T <: FloatNN | ComplexNN](output: Tensor[T], hx: Tensor[T])

case class TensorTriple[T <: FloatNN | ComplexNN](
    output: Tensor[T],
    indices: Tensor[T],
    values: Tensor[T]
)
