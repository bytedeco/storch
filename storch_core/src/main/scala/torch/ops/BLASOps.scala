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

package torch
package ops

import torch.nn.init.{NonLinearity, Mode}
import Layout.Strided
import Device.CPU
import internal.NativeConverters
import NativeConverters.*
import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.pytorch.{
  DoubleOptional,
  LongOptional,
  PackedSequence,
  Scalar,
  ScalarTypeOptional,
  ScalarOptional,
  TensorOptional,
  TensorOptionalList,
  TensorVector
}
import torch.internal.NativeConverters.{fromNative, toScalar}

import scala.collection.mutable.ListBuffer

/** BLAS and LAPACK Operations
  *
  * https://pytorch.org/docs/stable/torch.html#blas-and-lapack-operations
  */
private[torch] trait BLASOps {

  def nan_to_num[D <: DType](
      input: Tensor[D],
      nan: Option[Double] = None,
      posinf: Option[Double] = None,
      neginf: Option[Double] = None
  ): Tensor[D] =
    fromNative(
      torchNative.nan_to_num(input.native, toOptional(nan), toOptional(posinf), toOptional(neginf))
    )

  def scatter_reduce[D <: DType](
      input: Tensor[D],
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      src: Tensor[D],
      reduceMode: String,
      includeSelf: Boolean
  ): Tensor[D] = {
    index.dtype match
      case torch.int64 =>
        fromNative(
          torchNative.scatter_reduce(
            input.native,
            dim.toLong,
            index.native,
            src.native,
            reduceMode,
            includeSelf
          )
        )
      case torch.int32 =>
        fromNative(
          torchNative.scatter_reduce(
            input.native,
            dim.toLong,
            index.to(dtype = torch.int64).native,
            src.native,
            reduceMode,
            includeSelf
          )
        )
  }

  def scatter_add[D <: DType](
      input: Tensor[D],
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      src: Tensor[D]
  ): Tensor[D] = {
    index.dtype match
      case torch.int64 =>
        fromNative(torchNative.scatter_add(input.native, dim.toLong, index.native, src.native))
      case torch.int32 =>
        fromNative(
          torchNative.scatter_add(
            input.native,
            dim.toLong,
            index.to(dtype = torch.int64).native,
            src.native
          )
        )

  }

  /*
  // torch.scatter(input, dim, index, src) → Tensor
  //  def scatter[D <: DType](input: Tensor[D], dim: Int, index: Tensor[Int64], src: Tensor[D]) : Tensor[D]= {
  //    fromNative(torchNative.scatter(input.native, dim.toLong, index.native, src.native))
   */
  def scatter[D <: DType](
      input: Tensor[D],
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      src: Tensor[D],
      scatterMode: String
  ): Tensor[D] = {
    index.dtype match
      case torch.int64 =>
        fromNative(
          torchNative.scatter(input.native, dim.toLong, index.native, src.native, scatterMode)
        )
      case torch.int32 =>
        fromNative(
          torchNative.scatter(
            input.native,
            dim.toLong,
            index.to(dtype = torch.int64).native,
            src.native,
            scatterMode
          )
        )
  }

  def scatter[D <: DType, S <: ScalaType](
      input: Tensor[D],
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      src: S
  ): Tensor[D] = {
    index.dtype match
      case torch.int64 =>
        fromNative(torchNative.scatter(input.native, dim.toLong, index.native, toScalar(src)))
      case torch.int32 =>
        fromNative(
          torchNative.scatter(
            input.native,
            dim.toLong,
            index.to(dtype = torch.int64).native,
            toScalar(src)
          )
        )
  }

  def log_softmax[In <: DType, Out <: FloatNN | Derive](
      input: Tensor[In],
      dim: Int,
      dtype: Out = derive
  ): Tensor[DTypeOrDeriveFromTensor[In, Out]] =
    val derivedDType = dtype match
      case _: Derive => input.dtype
      case d: DType  => d
    val nativeDType =
      if dtype == input.dtype then ScalarTypeOptional()
      else ScalarTypeOptional(derivedDType.toScalarType)
    fromNative(torchNative.log_softmax(input.native, dim.toLong, nativeDType))

  def segment_reduce[D <: DType](input: Tensor[D], reduceMode: String): Tensor[D] = {
    fromNative(torchNative.segment_reduce(input.native, reduceMode))

  }

  /*
    public static native Tensor scatter(@Const @ByRef Tensor var0, @ByVal Dimname var1, @Const @ByRef Tensor var2, @Const @ByRef Tensor var3);
    public static native Tensor scatter(@Const @ByRef Tensor var0, @ByVal Dimname var1, @Const @ByRef Tensor var2, @Const @ByRef Scalar var3);
    public static native Tensor scatter(@Const @ByRef Tensor var0, @Cast({"int64_t"}) long var1, @Const @ByRef Tensor var3, @Const @ByRef Scalar var4, @StringView BytePointer var5);
  //public static native Tensor scatter(@Const @ByRef Tensor var0, @Cast({"int64_t"}) long var1, @Const @ByRef Tensor var3, @Const @ByRef Scalar var4, @StringView String var5);
  // torch.scatter(input, dim, index, src) → Tensor
   */
  def matmul[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    t1.matmul(t2)

  def dot[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    t1.dot(t2)

//    fromNative(
//    native.dot(other.native)
//  )

  // todo make sure the type of s is correct
  def vdot[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    t1.vdot(t2)

//    fromNative(
//    native.vdot(other.native)
//  )
  def put[D1 <: DType, D2 <: DType](
      input: Tensor[D1],
      index: Tensor[Int64],
      source: Tensor[D2],
      accumulate: Option[Boolean] = Some(true)
  ): Tensor[Promoted[D1, D2]] = {
    if accumulate.isDefined then
      fromNative(torchNative.put(input.native, index.native, source.native, accumulate.get))
    else fromNative(torchNative.put(input.native, index.native, source.native))
  }

  def bmm[D1 <: DType, D2 <: DType](input: Tensor[D1], mat2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.bmm(input.native, mat2.native))

  def bucketize[D1 <: DType, D2 <: DType](
      input: Tensor[D1],
      boundaries: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.bucketize(input.native, boundaries.native))

  def broadcast_to[D1 <: DType](input: Tensor[D1], shape: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.broadcast_to(input.native, shape*))

  def cartesian_prod[D1 <: DType](tensorArray: Seq[Tensor[D1]]): Tensor[D1] =
    val tensorVector = TensorVector(tensorArray.map(_.native).toArray*)
    fromNative(torchNative.cartesian_prod(tensorVector))

//  def cauchy[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.cauchy(t1.native))

  def ccol_indices_copy[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.ccol_indices_copy(t1.native))

  def cdist[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.cdist(t1.native, t2.native))

  def celu[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.celu(t1.native))

  def chain_matmul[D1 <: DType](tensorArray: Seq[Tensor[D1]]): Tensor[D1] =
    val tensorVector = TensorVector(tensorArray.map(_.native).toArray*)
    fromNative(torchNative.chain_matmul(tensorVector))

  def channel_shuffle[D1 <: DType](t1: Tensor[D1], sliceSeq: Long): Tensor[D1] =
    fromNative(torchNative.channel_shuffle(t1.native, sliceSeq))

  def cholesky[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.cholesky(t1.native))

  def cholesky_inverse[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.cholesky_inverse(t1.native))

  def cholesky_solve[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.cholesky_solve(t1.native, t2.native))

  def clamp_max[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.clamp_max(t1.native, t2.native))

  def clamp_min[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.clamp_min(t1.native, t2.native))

  def clip[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.clip(t1.native))

  def clip[D1 <: DType, S <: ScalaType](t1: Tensor[D1], min: S): Tensor[Div[D1, ScalaToDType[S]]] =
    fromNative(torchNative.clip(t1.native, new ScalarOptional(toScalar(min))))

  def clip[D1 <: DType, S <: ScalaType](
      t1: Tensor[D1],
      min: S,
      max: S
  ): Tensor[Div[D1, ScalaToDType[S]]] =
    fromNative(
      torchNative.clip(
        t1.native,
        new ScalarOptional(toScalar(min)),
        new ScalarOptional(toScalar(max))
      )
    )

  def clamp[D1 <: DType, S <: ScalaType](t1: Tensor[D1], min: S): Tensor[Div[D1, ScalaToDType[S]]] =
    fromNative(torchNative.clamp(t1.native, new ScalarOptional(toScalar(min))))

  def clamp[D1 <: DType, S <: ScalaType](
      t1: Tensor[D1],
      min: S,
      max: S
  ): Tensor[Div[D1, ScalaToDType[S]]] =
    fromNative(
      torchNative.clamp(
        t1.native,
        new ScalarOptional(toScalar(min)),
        new ScalarOptional(toScalar(max))
      )
    )

  def renorm[D1 <: DType](t1: Tensor[D1], p: Double, maxNorm: Double, dim: Int): Tensor[D1] =
    fromNative(torchNative.renorm(t1.native, toScalar(p), dim.toLong, toScalar(maxNorm)))

  def renorm[D1 <: DType](t1: Tensor[D1], p: Float, dim: Int, maxNorm: Float): Tensor[D1] =
    fromNative(torchNative.renorm(t1.native, toScalar(p), dim.toLong, toScalar(maxNorm)))

  def clamp_[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.clamp_(t1.native))

  def clip_[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.clip_(t1.native))

  def clip_[D1 <: DType, S <: ScalaType](t1: Tensor[D1], min: S): Tensor[Div[D1, ScalaToDType[S]]] =
    fromNative(torchNative.clip_(t1.native, new ScalarOptional(toScalar(min))))

  def clip_[D1 <: DType, S <: ScalaType](
      t1: Tensor[D1],
      min: S,
      max: S
  ): Tensor[Div[D1, ScalaToDType[S]]] =
    fromNative(
      torchNative.clip_(
        t1.native,
        new ScalarOptional(toScalar(min)),
        new ScalarOptional(toScalar(max))
      )
    )

  def clamp_[D1 <: DType, S <: ScalaType](
      t1: Tensor[D1],
      min: S
  ): Tensor[Div[D1, ScalaToDType[S]]] =
    fromNative(torchNative.clamp_(t1.native, new ScalarOptional(toScalar(min))))

  def clamp_[D1 <: DType, S <: ScalaType](
      t1: Tensor[D1],
      min: S,
      max: S
  ): Tensor[Div[D1, ScalaToDType[S]]] =
    fromNative(
      torchNative.clamp_(
        t1.native,
        new ScalarOptional(toScalar(min)),
        new ScalarOptional(toScalar(max))
      )
    )

  def clone[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.clone(t1.native))

  def copy[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.copy(t1.native, t2.native))

  def cumprod[D1 <: DType](t1: Tensor[D1], sliceSeq: Long): Tensor[D1] =
    fromNative(torchNative.cumprod(t1.native, sliceSeq))

  def column_stack[D1 <: DType](tensorArray: Seq[Tensor[D1]]): Tensor[D1] =
    val tensorVector = TensorVector(tensorArray.map(_.native).toArray*)
    fromNative(torchNative.column_stack(tensorVector))

  def combinations[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.combinations(t1.native))

  // convolution //concat
  def copysign_native[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.copysign(t1.native, t2.native))

  def corrcoef[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.corrcoef(t1.native))

//  def cosh[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.cosh(t1.native))

  def det[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.det(t1.native))

  def diag_embed[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.diag_embed(t1.native))

  def diagflat[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.diagflat(t1.native))

  def diagonal[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.diagonal(t1.native))

  def diagonal_scatter_raw[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.diagonal_scatter(t1.native, t2.native))

  def diff[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.diff(t1.native))

//  def digamma[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.digamma(t1.native))

//  def dist[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.dist(t1.native, t2.native))

//  def div[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.div(t1.native, t2.native))

  def divide[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.divide(t1.native, t2.native))

  def elu[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.elu(t1.native))

//  def erf[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.erf(t1.native))
//
//  def erfinv[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.erfinv(t1.native))

//  def exp[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.exp(t1.native))
//
//  def exp2[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.exp2(t1.native))

//  def expm1[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.expm1(t1.native))

//  def exponential_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.exponential(t1.native))

  def fft_fft[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.fft_fft(t1.native))

  def fft_fft2[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.fft_fft2(t1.native))

  def fft_fftn[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.fft_fftn(t1.native))

  def fft_fftshift[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.fft_fftshift(t1.native))

  def fft_hfft[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.fft_hfft(t1.native))

  def fft_hfft2[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.fft_hfft2(t1.native))

  def fft_hfftn[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.fft_hfftn(t1.native))

  def fft_ifft2[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.fft_ifft2(t1.native))

  def fft_ifftshift[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.fft_ifftshift(t1.native))

//  def fix[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.fix(t1.native))

  def fliplr[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.fliplr(t1.native))

  def flipud[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.flipud(t1.native))

//  def floor[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.floor(t1.native))

//  def ceil[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.ceil(t1.native))
//
//  def frac[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.frac(t1.native))

  //  public static native T_TensorTensor_T frexp(@Const @ByRef Tensor var0);

  def gelu[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.gelu(t1.native))

  def geqrf[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1]) =
    val tensorTuple = torchNative.geqrf(t1.native)
    (fromNative(tensorTuple.get0), fromNative(tensorTuple.get1))

  def glu[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.glu(t1.native))

  def hardshrink[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.hardshrink(t1.native))

  def hardsigmoid[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.hardsigmoid(t1.native))

  def hardswish[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.hardswish(t1.native))

  def hardtanh[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.hardtanh(t1.native))

  def histc[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.histc(t1.native))

  def histogram[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1]) =
    val tup = torchNative.histogram(t1.native)
    (fromNative(tup.get0), fromNative(tup.get1))

//  def imag[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.imag(t1.native))

  def int_repr[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.int_repr(t1.native))

  def inverse[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.inverse(t1.native))

  def isneginf[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.isneginf(t1.native))

  def isposinf[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.isposinf(t1.native))

  def leaky_relu[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.leaky_relu(t1.native))

//  def lgamma[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.lgamma(t1.native))

  def lift[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.lift(t1.native))

  def lift_fresh[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.lift_fresh(t1.native))

  def linalg_cholesky[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.linalg_cholesky(t1.native))

  def linalg_cholesky_ex[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1]) =
    val tup = torchNative.linalg_cholesky_ex(t1.native)
    (fromNative(tup.get0), fromNative(tup.get1))

  def flip[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.flip(t1.native, sliceSeq*))

  def frobenius_norm[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.frobenius_norm(t1.native, sliceSeq*))

  def group_norm[D1 <: DType](t1: Tensor[D1], sliceSeq: Long): Tensor[D1] =
    fromNative(torchNative.group_norm(t1.native, sliceSeq))

  def layer_norm[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.layer_norm(t1.native, sliceSeq*))

  def pad[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.pad(t1.native, sliceSeq*))

  def pad_sequence_raw[D1 <: DType](tensorArray: Seq[Tensor[D1]]): Tensor[D1] =
    val tensorVector = TensorVector(tensorArray.map(_.native).toArray*)
    fromNative(torchNative.pad_sequence(tensorVector))

  def pad_packed_sequence[D1 <: DType](packedSequence: PackedSequence): (Tensor[D1], Tensor[D1]) =
    val tensorVec = torchNative.pad_packed_sequence(packedSequence)
    (fromNative(tensorVec.get0()), fromNative(tensorVec.get1()))

  // torch.nn.utils.rnn.pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None)[source]
  def pad_packed_sequence[D1 <: DType](
      packedSequence: PackedSequence,
      batch_first: Boolean = false,
      padding_value: Double = 0.0,
      total_length: Option[Long] = None
  ): (Tensor[D1], Tensor[D1]) =
    val native_length =
      if total_length.isDefined then new LongOptional(total_length.get) else new LongOptional()
    val tensorVec =
      torchNative.pad_packed_sequence(packedSequence, batch_first, padding_value, native_length)
    (fromNative(tensorVec.get0()), fromNative(tensorVec.get1()))

//  def permute[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
//    fromNative(torchNative.permute(t1.native, sliceSeq *))

  def quantized_max_pool1d[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.quantized_max_pool1d(t1.native, sliceSeq*))

  def quantized_max_pool2d[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.quantized_max_pool2d(t1.native, sliceSeq*))

  def quantized_max_pool3d[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.quantized_max_pool3d(t1.native, sliceSeq*))

  def reflection_pad1d[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.reflection_pad1d(t1.native, sliceSeq*))

  def reflection_pad2d[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.reflection_pad2d(t1.native, sliceSeq*))

  def reflection_pad3d[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.reflection_pad3d(t1.native, sliceSeq*))

  def linalg_cond[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.linalg_cond(t1.native))

  def linalg_det[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.linalg_det(t1.native))

  def linalg_diagonal[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.linalg_diagonal(t1.native))

  def linalg_eig[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1]) =
    val tup = torchNative.linalg_eig(t1.native)
    (fromNative(tup.get0), fromNative(tup.get1))

  def linalg_eigh[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1]) =
    val tup = torchNative.linalg_eigh(t1.native)
    (fromNative(tup.get0), fromNative(tup.get1))

  def linalg_eigvals[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.linalg_eigvals(t1.native))

  def linalg_eigvalsh[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.linalg_eigvalsh(t1.native))

  def linalg_inv[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.linalg_inv(t1.native))

  def linalg_inv_ex[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1]) =
    val tup = torchNative.linalg_inv_ex(t1.native)
    (fromNative(tup.get0), fromNative(tup.get1))

  def linalg_ldl_factor[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1]) =
    val tup = torchNative.linalg_ldl_factor(t1.native)
    (fromNative(tup.get0), fromNative(tup.get1))

  def linalg_ldl_factor_ex[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1]) =
    val tup = torchNative.linalg_ldl_factor_ex(t1.native)
    (fromNative(tup.get0), fromNative(tup.get1))

  def linalg_lu[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1]) =
    val tup = torchNative.linalg_lu(t1.native)
    (fromNative(tup.get0), fromNative(tup.get1))

  def linalg_lu_factor[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1]) =
    val tup = torchNative.linalg_lu_factor(t1.native)
    (fromNative(tup.get0), fromNative(tup.get1))

  def linalg_lu_factor_ex[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1]) =
    val tup = torchNative.linalg_lu_factor_ex(t1.native)
    (fromNative(tup.get0), fromNative(tup.get1))

  def linalg_matrix_exp[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.linalg_matrix_exp(t1.native))

  def linalg_matrix_norm[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.linalg_matrix_norm(t1.native))

  def linalg_matrix_rank[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.linalg_matrix_rank(t1.native))

  def linalg_norm[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.linalg_norm(t1.native))

  def linalg_multi_dot[D1 <: DType](tensorArray: Seq[Tensor[D1]]): Tensor[D1] =
    val tensorVector = TensorVector(tensorArray.map(_.native).toArray*)
    fromNative(torchNative.linalg_multi_dot(tensorVector))

  def linalg_pinv[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.linalg_pinv(t1.native))

  def linalg_svdvals[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.linalg_svdvals(t1.native))

  def linalg_tensorinv[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.linalg_tensorinv(t1.native))

  def linalg_vander[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.linalg_vander(t1.native))

  def linalg_vector_norm[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.linalg_vector_norm(t1.native))

//  def log[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.log(t1.native))
//
//  def log10[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.log10(t1.native))

  def float_power[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.float_power(t1.native, t2.native))

  def floor_divide[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.floor_divide(t1.native, t2.native))

  def fmax[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.fmax(t1.native, t2.native))

  def fmin[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.fmin(t1.native, t2.native))

  def fmod_raw[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.fmod(t1.native, t2.native))

  def gcd[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.gcd(t1.native, t2.native))

  def ge[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.ge(t1.native, t2.native))

  def ger[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.ger(t1.native, t2.native))

  def greater[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.greater(t1.native, t2.native))

  def greater_equal[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.greater_equal(t1.native, t2.native))

  def gt[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.gt(t1.native, t2.native))

  def gt[D1 <: DType, S <: ScalaType](t1: Tensor[D1], other: S): Tensor[Div[D1, ScalaToDType[S]]] =
    fromNative(torchNative.gt(t1.native, toScalar(other)))

  def heaviside[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.heaviside(t1.native, t2.native))

  def hinge_embedding_loss[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.hinge_embedding_loss(t1.native, t2.native))

  def histogram[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): (Tensor[Promoted[D1, D2]], Tensor[Promoted[D1, D2]]) =
    val tup = torchNative.histogram(t1.native, t2.native)
    (fromNative(tup.get0), fromNative(tup.get1))

  def hspmm[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.hspmm(t1.native, t2.native))

  def huber_loss[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.huber_loss(t1.native, t2.native))

  def hypot_raw[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.hypot(t1.native, t2.native))

  def igamma_raw[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.igamma(t1.native, t2.native))

  def igammac_raw[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      inplace: Boolean = false
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.igammac(t1.native, t2.native))

  def inner[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.inner(t1.native, t2.native))

  def kl_div[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.kl_div(t1.native, t2.native))

  def kron[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.kron(t1.native, t2.native))

  def l1_loss[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.l1_loss(t1.native, t2.native))

  def lcm[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.lcm(t1.native, t2.native))

//  def ldexp[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.ldexp(t1.native, t2.native))

  def le[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.le(t1.native, t2.native))

  def less[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.less(t1.native, t2.native))

  def less_equal[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.less_equal(t1.native, t2.native))

  object linalg {
    def cross[D1 <: DType, D2 <: DType](
        t1: Tensor[D1],
        t2: Tensor[D2]
    ): Tensor[Promoted[D1, D2]] =
      fromNative(torchNative.linalg_cross(t1.native, t2.native))

    def householder_product[D1 <: DType, D2 <: DType](
        t1: Tensor[D1],
        t2: Tensor[D2]
    ): Tensor[Promoted[D1, D2]] =
      fromNative(torchNative.linalg_householder_product(t1.native, t2.native))

    def lstsq[D1 <: DType, D2 <: DType](
        t1: Tensor[D1],
        t2: Tensor[D2]
    ): (Tensor[Promoted[D1, D2]], Tensor[Promoted[D1, D2]]) =
      val tup = torchNative.linalg_lstsq(t1.native, t2.native)
      (fromNative(tup.get0), fromNative(tup.get1))

    def matmul[D1 <: DType, D2 <: DType](
        t1: Tensor[D1],
        t2: Tensor[D2]
    ): Tensor[Promoted[D1, D2]] =
      fromNative(torchNative.linalg_matmul(t1.native, t2.native))

    def pinv[D1 <: DType, D2 <: DType](
        t1: Tensor[D1],
        t2: Tensor[D2]
    ): Tensor[Promoted[D1, D2]] =
      fromNative(torchNative.linalg_pinv(t1.native, t2.native))

    def solve[D1 <: DType, D2 <: DType](
        t1: Tensor[D1],
        t2: Tensor[D2]
    ): Tensor[Promoted[D1, D2]] =
      fromNative(torchNative.linalg_solve(t1.native, t2.native))

    def tensorsolve[D1 <: DType, D2 <: DType](
        t1: Tensor[D1],
        t2: Tensor[D2]
    ): Tensor[Promoted[D1, D2]] =
      fromNative(torchNative.linalg_tensorsolve(t1.native, t2.native))

    def vecdot[D1 <: DType, D2 <: DType](
        t1: Tensor[D1],
        t2: Tensor[D2]
    ): Tensor[Promoted[D1, D2]] =
      fromNative(torchNative.linalg_vecdot(t1.native, t2.native))

  }
  def linalg_cross[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.linalg_cross(t1.native, t2.native))

  def linalg_householder_product[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.linalg_householder_product(t1.native, t2.native))

  def linalg_lstsq[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): (Tensor[Promoted[D1, D2]], Tensor[Promoted[D1, D2]]) =
    val tup = torchNative.linalg_lstsq(t1.native, t2.native)
    (fromNative(tup.get0), fromNative(tup.get1))

  def linalg_matmul[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.linalg_matmul(t1.native, t2.native))

  def linalg_pinv[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.linalg_pinv(t1.native, t2.native))

  def linalg_solve[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.linalg_solve(t1.native, t2.native))

  def linalg_tensorsolve[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.linalg_tensorsolve(t1.native, t2.native))

  def linalg_vecdot[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.linalg_vecdot(t1.native, t2.native))

  def linear_raw[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.linear(t1.native, t2.native))

//  def logaddexp[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.logaddexp(t1.native, t2.native))

  def logical_and[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.logical_and(t1.native, t2.native))

  def logical_or[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.logical_or(t1.native, t2.native))

  def logical_xor[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.logical_xor(t1.native, t2.native))

  def count_nonzero_raw[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.count_nonzero(t1.native, sliceSeq*))

  def dequantize[D1 <: DType](tensor: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.dequantize(tensor.native))

  def cov[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.cov(t1.native))

//  def deg2rad[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.deg2rad(t1.native))

  def cumulative_trapezoid[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.cumulative_trapezoid(t1.native))

  def cross[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.cross(t1.native, t2.native))

  def cross_entropy_loss[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.cross_entropy_loss(t1.native, t2.native))

  def conv_tbc[D1 <: DType, D2 <: DType, D3 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      t3: Tensor[D1 | D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.conv_tbc(t1.native, t2.native, t3.native))

  def linalg_ldl_solve[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      t3: Tensor[D1 | D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.linalg_ldl_solve(t1.native, t2.native, t3.native))

//  def lerp[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2], t3: Tensor[D1 | D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.lerp(t1.native, t2.native, t3.native))

  def linalg_lu_solve[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      t3: Tensor[D1 | D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.linalg_lu_solve(t1.native, t2.native, t3.native))

  def lu_solve[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      t3: Tensor[D1 | D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.lu_solve(t1.native, t2.native, t3.native))

  def margin_ranking_loss[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      t3: Tensor[D1 | D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.margin_ranking_loss(t1.native, t2.native, t3.native))

  def ormqr[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      t3: Tensor[D1 | D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.ormqr(t1.native, t2.native, t3.native))

  def cosine_embedding_loss[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      t3: Tensor[D1 | D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.cosine_embedding_loss(t1.native, t2.native, t3.native))

//  def log1p[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.log1p(t1.native))
//
//  def log2[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.log2(t1.native))

  def log_normal_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.log_normal(t1.native))

  def log_sigmoid[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.log_sigmoid(t1.native))

  def logdet[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.logdet(t1.native))

  def logical_not[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.logical_not(t1.native))

  def lt[D1 <: DType, S <: ScalaType](t1: Tensor[D1], other: S): Tensor[Div[D1, ScalaToDType[S]]] =
    fromNative(torchNative.lt(t1.native, toScalar(other)))

  def lt[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.lt(t1.native, t2.native))

  def maximum[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.maximum(t1.native, t2.native))

  def minimum[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.minimum(t1.native, t2.native))

//  def min[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.min(t1.native, t2.native))

  def outer[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.outer(t1.native, t2.native))

  def pairwise_distance[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.pairwise_distance(t1.native, t2.native))

  def special_zeta[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.special_zeta(t1.native, t2.native))

  def special_xlogy[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.special_xlogy(t1.native, t2.native))

  def special_xlog1py[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.special_xlog1py(t1.native, t2.native))

  def special_shifted_chebyshev_polynomial_w[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.special_shifted_chebyshev_polynomial_w(t1.native, t2.native))

  def special_shifted_chebyshev_polynomial_v[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.special_shifted_chebyshev_polynomial_v(t1.native, t2.native))

  def special_shifted_chebyshev_polynomial_u[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.special_shifted_chebyshev_polynomial_u(t1.native, t2.native))

  def special_shifted_chebyshev_polynomial_t[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.special_shifted_chebyshev_polynomial_t(t1.native, t2.native))

  def special_hermite_polynomial_he[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.special_hermite_polynomial_he(t1.native, t2.native))

  def special_hermite_polynomial_h[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.special_hermite_polynomial_h(t1.native, t2.native))

  def special_gammainc[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.special_gammainc(t1.native, t2.native))

  def special_chebyshev_polynomial_w[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.special_chebyshev_polynomial_w(t1.native, t2.native))

  def special_chebyshev_polynomial_v[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.special_chebyshev_polynomial_v(t1.native, t2.native))

  def special_chebyshev_polynomial_u[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.special_chebyshev_polynomial_u(t1.native, t2.native))

  def special_chebyshev_polynomial_t[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.special_chebyshev_polynomial_t(t1.native, t2.native))

  def soft_margin_loss[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.soft_margin_loss(t1.native, t2.native))

  def smooth_l1_loss[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.smooth_l1_loss(t1.native, t2.native))

  def smm[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.smm(t1.native, t2.native))

  def slice_scatter_raw[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.slice_scatter(t1.native, t2.native))

  def slice_inverse[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.slice_inverse(t1.native, t2.native))

  def set[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.set(t1.native, t2.native))

  def searchsorted[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.searchsorted(t1.native, t2.native))

  def rsub[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.rsub(t1.native, t2.native))

  def rrelu_with_noise_raw[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.rrelu_with_noise(t1.native, t2.native))

  def resize_as_sparse[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.resize_as_sparse(t1.native, t2.native))

  def resize_as_[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.resize_as_(t1.native, t2.native))

//  def remainder[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.remainder(t1.native, t2.native))

  def quantile[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.quantile(t1.native, t2.native))

  def prelu[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.prelu(t1.native, t2.native))

//  def pow[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.pow(t1.native, t2.native))

  def polar[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.polar(t1.native, t2.native))

  def orgqr[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.orgqr(t1.native, t2.native))

  def not_equal[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.not_equal(t1.native, t2.native))

  def nll_loss_nd[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.nll_loss_nd(t1.native, t2.native))

  def nll_loss2d[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.nll_loss2d(t1.native, t2.native))

  def nll_loss[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.nll_loss(t1.native, t2.native))

  def ne[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.ne(t1.native, t2.native))

  def nextafter_raw[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      inplace: Boolean = false
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.nextafter(t1.native, t2.native))

  def nanquantile[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.nanquantile(t1.native, t2.native))

  def mv[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.mv(t1.native, t2.native))

  def multiply[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.multiply(t1.native, t2.native))

  def multilabel_margin_loss[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.multilabel_margin_loss(t1.native, t2.native))

  def multi_margin_loss[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.multi_margin_loss(t1.native, t2.native))

  def mse_loss[D1 <: DType, D2 <: DType](
      input: Tensor[D1],
      target: Tensor[D2],
      reduction: Long
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.mse_loss(input.native, target.native, reduction))

  def mse_loss[D1 <: DType, D2 <: DType](
      input: Tensor[D1],
      target: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.mse_loss(input.native, target.native))

  def mm[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] = {
    if t1.dim > 2 then fromNative(torchNative.matmul(t1.native, t2.native))
    else fromNative(torchNative.mm(t1.native, t2.native))
  }

  def mkldnn_linear[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.mkldnn_linear(t1.native, t2.native))

  def matrix_exp[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.matrix_exp(t1.native))

//  def max[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.max(t1.native))

  def mean_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.mean(t1.native))

//  def median[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.median(t1.native))

  def clip_grad_norm_(
      parameters: Seq[Tensor[?]],
      max_norm: Double
  ): Double =
    torchNative.clip_grad_norm_(
      TensorVector(parameters.map(_.native).toArray*),
      max_norm
    )

  def clip_grad_norm_(
      parameters: Tensor[?],
      max_norm: Double
  ): Double =
    torchNative.clip_grad_norm_(
      parameters.native,
      max_norm
    )

  def clip_grad_norm_(
      parameters: Tensor[?],
      max_norm: Double,
      norm_type: Double = 2.0,
      error_if_nonfinite: Boolean = false
  ): Double =
    torchNative.clip_grad_norm_(
      parameters.native,
      max_norm,
      norm_type,
      error_if_nonfinite
    )

  //  norm_type: Double = 2.0,
  //                       error_if_nonfinite: Boolean = false
  def clip_grad_norm_(
      parameters: Seq[Tensor[?]],
      max_norm: Double,
      norm_type: Double,
      error_if_nonfinite: Boolean
  ): Double =
    torchNative.clip_grad_norm_(
      TensorVector(parameters.map(_.native).toArray*),
      max_norm,
      norm_type,
      error_if_nonfinite
    )

  /** *
    *
    * @param parameters
    * @param clip_value
    * @return
    */
  // torch.nn.utils.clip_grad_value_(parameters, clip_value, foreach=None)
  def clip_grad_value_(
      parameters: Seq[Tensor[?]],
      clip_value: Double
  ): Unit =
    torchNative.clip_grad_value_(
      TensorVector(parameters.map(_.native).toArray*),
      clip_value
    )

  def clip_grad_value_(
      parameters: Tensor[?],
      clip_value: Double
  ): Unit =
    torchNative.clip_grad_value_(
      parameters.native,
      clip_value
    )

  def gammainc[D1 <: DType, D2 <: DType](
      input: Tensor[D1],
      other: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.gammainc(input.native, other.native))

  def gammaincc[D1 <: DType, D2 <: DType](
      input: Tensor[D1],
      other: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.gammaincc(input.native, other.native))

  def parameters_to_vector(
      parameters: Seq[Tensor[?]]
  ): Tensor[?] = {
    val tensor = torchNative.parameters_to_vector(
      TensorVector(parameters.map(_.native).toArray*)
    )
    fromNative(tensor)
  }

  def vector_to_parameters(
      vec: Tensor[?],
      parameters: Seq[Tensor[?]]
  ): Unit =
    torchNative.vector_to_parameters(
      vec.native,
      TensorVector(parameters.map(_.native).toArray*)
    )

  //  def min[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.min(t1.native))

  def mish[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.mish(t1.native))

  def mkldnn_reorder_conv2d_weight[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.mkldnn_reorder_conv2d_weight(t1.native))

  def mkldnn_reorder_conv3d_weight[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.mkldnn_reorder_conv3d_weight(t1.native))

  def sort[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1]) =
    val tup = torchNative.sort(t1.native)
    (fromNative(tup.get0), fromNative(tup.get1))

  def msort[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.msort(t1.native))

  def nan_to_num[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.nan_to_num(t1.native))

  def nanmean_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.nanmean(t1.native))

//  def nanmedian[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.nanmedian(t1.native))

//  def nansum[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.nansum(t1.native))

  def native_norm[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.native_norm(t1.native))

//  def neg[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.neg(t1.native))

  def negative[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.negative(t1.native))

//  def nonzero[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.nonzero(t1.native))

  def expand_size[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    val tensors = torchNative.expand_size(t1.native, sliceSeq*)
    fromNative(tensors.access)

  def expand_inplace[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    val tensors = torchNative.expand_inplace(t1.native, t2.native)
    fromNative(tensors.access)

  def adaptive_max_pool3d[D1 <: DType](
      t1: Tensor[D1],
      sliceSeq: Seq[Long]
  ): (Tensor[D1], Tensor[D1]) =
    val tup = torchNative.adaptive_max_pool3d(t1.native, sliceSeq*)
    (fromNative(tup.get0), fromNative(tup.get1))

  def adaptive_max_pool2d[D1 <: DType](
      t1: Tensor[D1],
      sliceSeq: Seq[Long]
  ): (Tensor[D1], Tensor[D1]) =
    val tup = torchNative.adaptive_max_pool2d(t1.native, sliceSeq*)
    (fromNative(tup.get0), fromNative(tup.get1))

  def adaptive_max_pool1d[D1 <: DType](
      t1: Tensor[D1],
      sliceSeq: Seq[Long]
  ): (Tensor[D1], Tensor[D1]) =
    val tup = torchNative.adaptive_max_pool1d(t1.native, sliceSeq*)
    (fromNative(tup.get0), fromNative(tup.get1))

  // >>> from torch.nn.utils.rnn import pack_sequence
  // >>> a = torch.tensor([1, 2, 3])
  // >>> b = torch.tensor([4, 5])
  // >>> c = torch.tensor([6])
  // >>> pack_sequence([a, b, c])
  // PackedSequence(data=tensor([1, 4, 6, 2, 5, 3]), batch_sizes=tensor([3, 2, 1]), sorted_indices=None, unsorted_indices=None)
  // torch.nn.utils.rnn.pack_sequence(sequences, enforce_sorted=True)
  def pack_sequence[D1 <: DType](sequences: Seq[Tensor[D1]]): PackedSequence =
    val tensorVector = TensorVector(sequences.map(_.native).toArray*)
    val packSeq = torchNative.pack_sequence(tensorVector)
    packSeq

  def pack_sequence[D1 <: DType](
      sequences: Seq[Tensor[D1]],
      enforce_sorted: Boolean = true
  ): PackedSequence =
    val tensorVector = TensorVector(sequences.map(_.native).toArray*)
    val packSeq = torchNative.pack_sequence(tensorVector, enforce_sorted)
    packSeq

  /** *
    *
    * @param input
    * @param lengths
    * @return
    */
  // torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True)
  def pack_padded_sequence[D1 <: DType, D2 <: DType](
      input: Tensor[D1],
      lengths: Tensor[D2]
  ): PackedSequence =
    val packPadSeq = torchNative.pack_padded_sequence(input.native, lengths.native)
    packPadSeq

  def pack_padded_sequence[D1 <: DType, D2 <: DType](
      input: Tensor[D1],
      lengths: Tensor[D2],
      batch_first: Boolean = false,
      enforce_sorted: Boolean = true
  ): PackedSequence =
    val packPadSeq =
      torchNative.pack_padded_sequence(input.native, lengths.native, batch_first, enforce_sorted)
    packPadSeq

  def nonzero_numpy[D1 <: DType](t1: Tensor[D1]): Seq[Tensor[D1]] =
    val tensorArray = torchNative.nonzero_numpy(t1.native)
    tensorVectorToSeqTensor(tensorArray)

  def tensorVectorToSeqTensor[D1 <: DType](vec: TensorVector): Seq[Tensor[D1]] = {
    var it = vec.begin()
    val tensorSeq = new ListBuffer[Tensor[D1]]()
    while (!it.equals(vec.end())) {
      val tensor = fromNative(it.get()).asInstanceOf[Tensor[D1]]
      tensorSeq.append(tensor)
      it = it.increment()
    }
    tensorSeq.toSeq
  }

//  def hsplit[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Seq[Tensor[D1]] =
//    val tensorRawSeq: TensorVector = torchNative.hsplit(t1.native, sliceSeq *)
//    tensorVectorToSeqTensor(tensorRawSeq)

//  def vsplit[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Seq[Tensor[D1]] =
//    val tensorRawSeq: TensorVector = torchNative.vsplit(t1.native, sliceSeq *)
//    tensorVectorToSeqTensor(tensorRawSeq)

//  def where[D1 <: DType](t1: Tensor[D1]): Seq[Tensor[D1]] =
//    val tensorArray: TensorVector = torchNative.where(t1.native)
//    tensorVectorToSeqTensor(tensorArray)

  def broadcast_tensors[D1 <: DType](tensorArray: Seq[Tensor[D1]]): Seq[Tensor[D1]] =
    val tensorVector = TensorVector(tensorArray.map(_.native).toArray*)
    val broadTensorArray: TensorVector = torchNative.broadcast_tensors(tensorVector)
    tensorVectorToSeqTensor(broadTensorArray)

  def meshgrid[D1 <: DType](tensorArray: Seq[Tensor[D1]]): Seq[Tensor[D1]] =
    val tensorVector = TensorVector(tensorArray.map(_.native).toArray*)
    val tensorArr: TensorVector = torchNative.meshgrid(tensorVector)
    tensorVectorToSeqTensor(tensorArr)

  def dequantize[D1 <: DType](tensorArray: Seq[Tensor[D1]]): Seq[Tensor[D1]] =
    val tensorVector = TensorVector(tensorArray.map(_.native).toArray*)
    val tensorArr: TensorVector = torchNative.dequantize(tensorVector)
    tensorVectorToSeqTensor(tensorArr)

  def norm[D1 <: FloatNN | BFloat16](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.norm(t1.native))

  def norm[D1 <: FloatNN | BFloat16, S <: ScalaType](t1: Tensor[D1], p: S): Tensor[D1] = {
    val pFloat = p match {
      case m: Float  => m
      case m: Double => m.toFloat
      case m: Int    => m.toFloat
      case m: Long   => m.toFloat
    }
    fromNative(torchNative.norm(t1.native, ScalarOptional(toScalar(pFloat))))
  }

  def norm[D1 <: FloatNN | BFloat16, S <: ScalaType](
      t1: Tensor[D1],
      p: S,
      dim: Int*
  ): Tensor[D1] = {
    val pFloat = p match {
      case m: Float  => m
      case m: Double => m.toFloat
      case m: Int    => m.toFloat
      case m: Long   => m.toFloat
    }
    fromNative(torchNative.norm(t1.native, ScalarOptional(toScalar(pFloat)), dim.map(_.toLong)*))
  }

  def norm[D1 <: FloatNN | BFloat16, S <: ScalaType](
      t1: Tensor[D1],
      p: S,
      dim: Seq[Long],
      keepdim: Boolean = false
  ): Tensor[D1] = {
    val pFloat = p match {
      case m: Float  => m
      case m: Double => m.toFloat
      case m: Int    => m.toFloat
      case m: Long   => m.toFloat
    }
    fromNative(torchNative.norm(t1.native, ScalarOptional(toScalar(pFloat)), dim.toArray, keepdim))
  }

  def norm_except_dim[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.norm_except_dim(t1.native))

  def normal_functional[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.normal_functional(t1.native))

  def normal_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.normal(t1.native))

  def nuclear_norm[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.nuclear_norm(t1.native))

//  def ones[D1 <: DType](t1: Seq[Long]): Tensor[D1] =
//    fromNative(torchNative.ones(t1 *))

//  def ones_like[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.ones_like(t1.native))

  def pdist[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.pdist(t1.native))

  def pinverse[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.pinverse(t1.native))

  def poisson_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.poisson(t1.native))

//  def positive[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.positive(t1.native))

//  def prod[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.prod(t1.native))

  def replication_pad1d[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.replication_pad1d(t1.native, sliceSeq*))

  def replication_pad2d[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.replication_pad2d(t1.native, sliceSeq*))

  def replication_pad3d[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.replication_pad3d(t1.native, sliceSeq*))

//  def reshape[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
//    fromNative(torchNative.reshape(t1.native, sliceSeq *))

  def resize[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.resize(t1.native, sliceSeq*))

  def q_per_channel_zero_points[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.q_per_channel_zero_points(t1.native))

  def qr[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1]) =
    val tensorTuple = torchNative.qr(t1.native)
    (fromNative(tensorTuple.get0), fromNative(tensorTuple.get1))

  def rad2deg[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.rad2deg(t1.native))

  def rand_like_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.rand_like(t1.native))

  def randint_like[D1 <: DType](t1: Tensor[D1], ln: Long): Tensor[D1] =
    fromNative(torchNative.randint_like(t1.native, ln))

  def randn_like[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.randn_like(t1.native))

  def random[D1 <: DType](t1: Tensor[D1], seed: Long): Tensor[D1] =
    fromNative(torchNative.random(t1.native, seed))

  def random[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.random(t1.native))

  def ravel[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.ravel(t1.native))

  def kaiming_normal_[D <: DType](
      weight: Tensor[D],
      a: Double = 0,
      mode: Mode = Mode.FanIn,
      nonlinearity: NonLinearity = NonLinearity.LeakyReLU
  ): weight.type =
    torchNative.kaiming_normal_(weight.native, a, mode.toNative, nonlinearity.toNative)
    weight

  def kaiming_uniform_[D <: DType](
      weight: Tensor[D],
      a: Double = 0,
      mode: Mode = Mode.FanIn,
      nonlinearity: NonLinearity = NonLinearity.LeakyReLU
  ): weight.type =
    torchNative.kaiming_uniform_(weight.native, a, mode.toNative, nonlinearity.toNative)
    weight

  def xavier_uniform_[D <: DType](
      weight: Tensor[D],
      gain: Double = 1.0
  ): weight.type =
    torchNative.xavier_uniform_(weight.native, gain)
    weight

  def xavier_normal_[D <: DType](
      weight: Tensor[D],
      gain: Double = 1.0
  ): weight.type =
    torchNative.xavier_normal_(weight.native, gain)
    weight

  def normal_[D <: DType](
      weight: Tensor[D],
      mean: Double = 0,
      std: Double = 0
  ): weight.type =
    torchNative.normal_(weight.native, mean, std)
    weight

  def uniform_[D <: DType](
      weight: Tensor[D],
      a: Double = 0,
      b: Double = 1
  ): Tensor[D] =
    torchNative.uniform_(weight.native, a, b)
    weight

  def calculate_gain(
      nonlinearity: NonLinearity = NonLinearity.LeakyReLU,
      param: Double = 0.01
  ): Double =
    torchNative.calculate_gain(nonlinearity.toNative, param)

  def constant_[D <: DType](weight: Tensor[D], fillValue: Double): weight.type =
    torchNative.constant_(weight.native, Scalar(fillValue))
    weight

  def ones_[D <: DType](
      weight: Tensor[D]
  ): weight.type =
    torchNative.ones_(weight.native)
    weight

  def zeros_[D <: DType](
      weight: Tensor[D]
  ): weight.type =
    torchNative.zeros_(weight.native)
    weight

  def eye_[D <: DType](
      weight: Tensor[D]
  ): weight.type =
    torchNative.eye_(weight.native)
    weight

  def dirac_[D <: DType](
      weight: Tensor[D]
  ): weight.type =
    torchNative.dirac_(weight.native)
    weight

  //    public static native Tensor repeat_interleave(@Const @ByRef Tensor var0, LongOptional var1);

  //    public static native Tensor repeat_interleave(Tensor var0, Tensor var1, LongOptional var2,  LongOptional var3);
  def repeat_interleave[D1 <: DType](
      t1: Tensor[D1],
      repeats: Tensor[Int64],
      dim: Option[Long],
      output_size: Option[Long] = None
  ): Tensor[D1] = {
    val dimOpt = if dim.isDefined then new LongOptional(dim.get) else new LongOptional()
    val outputSizeOpt =
      if output_size.isDefined then new LongOptional(output_size.get) else new LongOptional()
    fromNative(torchNative.repeat_interleave(t1.native, repeats.native, dimOpt, outputSizeOpt))
  }

  //    public static native Tensor repeat_interleave(@Const @ByRef Tensor var0, @Cast({"int64_t"}) long var1, LongOptional var3, LongOptional var4);
  def repeat_interleave[D1 <: DType](
      t1: Tensor[D1],
      repeats: Long,
      dim: Option[Long],
      output_size: Option[Long]
  ): Tensor[D1] = {
    val dimOpt = if dim.isDefined then new LongOptional(dim.get) else new LongOptional()
    val outputSizeOpt =
      if output_size.isDefined then new LongOptional(output_size.get) else new LongOptional()
    fromNative(torchNative.repeat_interleave(t1.native, repeats, dimOpt, outputSizeOpt))
  }

  def repeat_interleave[D1 <: DType](t1: Tensor[D1], repeats: Int, dim: Int): Tensor[D1] =
    fromNative(
      torchNative.repeat_interleave(t1.native, repeats.toLong, new LongOptional(dim.toLong), new LongOptional())
    )

  def repeat_interleave[D1 <: DType](t1: Tensor[D1], repeats: Int): Tensor[D1] =
    fromNative(torchNative.repeat_interleave(t1.native, repeats.toLong))

  def repeat_interleave[D1 <: DType](t1: Tensor[D1], repeats: Tensor[Int64]): Tensor[D1] =
    fromNative(torchNative.repeat_interleave(t1.native, repeats.native))

  def repeat_interleave[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.repeat_interleave(t1.native))

//  def repeat_interleave[D1 <: DType](t1: Long): Tensor[D1] =
//    fromNative(torchNative.repeat_interleave(t1))

  def relu[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.relu(t1.native))

  def relu6[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.relu6(t1.native))

  def resolve_conj[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.resolve_conj(t1.native))

  def resolve_neg[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.resolve_neg(t1.native))

  def rms_norm[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.rms_norm(t1.native, sliceSeq*))

  def rms_norm[D1 <: DType](
      t1: Tensor[D1],
      normalizedShape: Seq[Long],
      weight: Option[Tensor[Float32]],
      eps: Option[Double]
  ): Tensor[D1] = {
    val weightOpt =
      if weight.isDefined then new TensorOptional(weight.get.native) else new TensorOptional()
    val epsOpt = if eps.isDefined then new DoubleOptional(eps.get) else new DoubleOptional()
    fromNative(torchNative.rms_norm(t1.native, normalizedShape.toArray, weightOpt, epsOpt))
  }

  def roll[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.roll(t1.native, sliceSeq*))

  def special_logsumexp[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.special_logsumexp(t1.native, sliceSeq*))

//  def tile[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
//    fromNative(torchNative.tile(t1.native, sliceSeq *))

//  def vstack[D1 <: DType](tensorArray: Seq[Tensor[D1]]): Tensor[D1] =
//    val tensorVector = TensorVector(tensorArray.map(_.native).toArray *)
//    fromNative(torchNative.vstack(tensorVector))

  def rot90[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.rot90(t1.native))

//  def round[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.round(t1.native))

  def row_indices_copy[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.row_indices_copy(t1.native))

  def row_stack[D1 <: DType](tensorArray: Seq[Tensor[D1]]): Tensor[D1] =
    val tensorVector = TensorVector(tensorArray.map(_.native).toArray*)
    fromNative(torchNative.row_stack(tensorVector))

  def rrelu_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.rrelu(t1.native))

//  def rsqrt[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.rsqrt(t1.native))

  def selu[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.selu(t1.native))

//  def sgn[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.sgn(t1.native))
//
//  def sign[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.sign(t1.native))

//  def signbit[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.signbit(t1.native))

  def silu[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.silu(t1.native))

//  def sinc[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.sinc(t1.native))

//  def sinh[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.sinh(t1.native))

  def slice[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.slice(t1.native))

  def slice_copy[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.slice_copy(t1.native))

  def softplus[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.softplus(t1.native))

  def softshrink[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.softshrink(t1.native))

  def invert_permutation[D <: DType](permutation: Tensor[D]): Tensor[D] = {

    fromNative(torchNative.invert_permutation(permutation.native))
  }

  def special_airy_ai[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_airy_ai(t1.native))

  def special_bessel_j0[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_bessel_j0(t1.native))

  def special_bessel_y0[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_bessel_y0(t1.native))

  def special_bessel_y1[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_bessel_y1(t1.native))

  def special_i0[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_i0(t1.native))

  def special_gammaln[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_gammaln(t1.native))

  def special_expm1[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_expm1(t1.native))

  def special_expit[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_expit(t1.native))

  def special_exp2[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_exp2(t1.native))

  def special_erfinv[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_erfinv(t1.native))

  def special_erfcx[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_erfcx(t1.native))

  def special_erfc[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_erfc(t1.native))

  def special_erf[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_erf(t1.native))

  def special_entr[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_entr(t1.native))

  def special_digamma[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_digamma(t1.native))

  def special_i1e[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_i1e(t1.native))

  def special_i1[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_i1(t1.native))

  def special_log1p[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_log1p(t1.native))

  def special_log_ndtr[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_log_ndtr(t1.native))

  def special_logit[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_logit(t1.native))

  def special_modified_bessel_i0[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_modified_bessel_i0(t1.native))

  def special_modified_bessel_i1[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_modified_bessel_i1(t1.native))

  def special_modified_bessel_k0[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_modified_bessel_k0(t1.native))

  def special_modified_bessel_k1[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_modified_bessel_k1(t1.native))

  def special_ndtr[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_ndtr(t1.native))

  def special_ndtri[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_ndtri(t1.native))

  def special_psi[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_psi(t1.native))

  def special_round[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_round(t1.native))

  def special_scaled_modified_bessel_k0[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_scaled_modified_bessel_k0(t1.native))

  def special_scaled_modified_bessel_k1[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_scaled_modified_bessel_k1(t1.native))

  def special_sinc[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_sinc(t1.native))

  def special_spherical_bessel_j0[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.special_spherical_bessel_j0(t1.native))

  def xlogy_native[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.xlogy(t1.native, t2.native))

  def embedding_renorm[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      d1: Double,
      d2: Double
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.embedding_renorm(t1.native, t2.native, d1, d2))

  def embedding_renorm_[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      d1: Double,
      d2: Double
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.embedding_renorm_(t1.native, t2.native, d1, d2))

  def copy_sparse_to_sparse[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.copy_sparse_to_sparse(t1.native, t2.native))

  def arctan2[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.arctan2(t1.native, t2.native))

  def binary_cross_entropy[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.binary_cross_entropy(t1.native, t2.native))

  def binary_cross_entropy_with_logits[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.binary_cross_entropy_with_logits(t1.native, t2.native))

  def binomial_raw[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.binomial(t1.native, t2.native))

  def bitwise_and[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.bitwise_and(t1.native, t2.native))

  def bitwise_not[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.bitwise_not(t1.native))

  def bitwise_or[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.bitwise_or(t1.native, t2.native))

  def bitwise_xor[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.bitwise_xor(t1.native, t2.native))

  def notEquals[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.notEquals(t1.native, t2.native))

  def !=[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.notEquals(t1.native, t2.native))

  def bitwise_or[D1 <: DType, S <: ScalaType](
      t1: Tensor[D1],
      t2: S
  ): Tensor[Div[D1, ScalaToDType[S]]] =
    fromNative(torchNative.bitwise_or(t1.native, toScalar(t2)))

  def bitwise_xor[D1 <: DType, S <: ScalaType](
      t1: Tensor[D1],
      t2: S
  ): Tensor[Div[D1, ScalaToDType[S]]] =
    fromNative(torchNative.bitwise_xor(t1.native, toScalar(t2)))

  def notEquals[D1 <: DType, S <: ScalaType](
      t1: Tensor[D1],
      t2: S
  ): Tensor[Div[D1, ScalaToDType[S]]] =
    fromNative(torchNative.notEquals(t1.native, toScalar(t2)))

  def !=[D1 <: DType, S <: ScalaType](
      t1: Tensor[D1],
      t2: S
  ): Tensor[Div[D1, ScalaToDType[S]]] =
    fromNative(torchNative.notEquals(t1.native, toScalar(t2)))

  def bitwise_or[S <: ScalaType, D2 <: DType](
      t1: S,
      t2: Tensor[D2]
  ): Tensor[Div[D2, ScalaToDType[S]]] =
    fromNative(torchNative.bitwise_or(toScalar(t1), t2.native))

  def bitwise_xor[S <: ScalaType, D2 <: DType](
      t1: S,
      t2: Tensor[D2]
  ): Tensor[Div[D2, ScalaToDType[S]]] =
    fromNative(torchNative.bitwise_xor(toScalar(t1), t2.native))

  def notEquals[S <: ScalaType, D2 <: DType](
      t1: S,
      t2: Tensor[D2]
  ): Tensor[Div[D2, ScalaToDType[S]]] =
    fromNative(torchNative.notEquals(toScalar(t1), t2.native))

  def !=[S <: ScalaType, D2 <: DType](
      t1: S,
      t2: Tensor[D2]
  ): Tensor[Div[D2, ScalaToDType[S]]] =
    fromNative(torchNative.notEquals(toScalar(t1), t2.native))

  def equal[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Boolean =
    torchNative.equal(t1.native, t2.native)

  def equals[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.equals(t1.native, t2.native))

  def equals[D1 <: DType, S <: ScalaType](
      t1: Tensor[D1],
      other: S
  ): Tensor[Div[D1, ScalaToDType[S]]] =
    fromNative(torchNative.equals(t1.native, toScalar(other)))

  def equals[D2 <: DType, S <: ScalaType](
      t1: S,
      other: Tensor[D2]
  ): Tensor[Div[D2, ScalaToDType[S]]] =
    fromNative(torchNative.equals(toScalar(t1), other.native))

  def eq[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.eq(t1.native, t2.native))

  def greaterThanEquals[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.greaterThanEquals(t1.native, t2.native))

  def greaterThan[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.greaterThan(t1.native, t2.native))

  def lessThanEquals[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.lessThanEquals(t1.native, t2.native))

  def lessThan[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.lessThan(t1.native, t2.native))

  def xor[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.xor(t1.native, t2.native))

  def or[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.or(t1.native, t2.native))

  def and[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.and(t1.native, t2.native))

  def mod[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.mod(t1.native, t2.native))

//  def eq[S <: ScalaType, D2 <: DType](t1: S, t2: Tensor[D2]): Tensor[Promoted[S, D2]] =
//    fromNative(torchNative.eq(toScalar(t1), t2.native))

  def greaterThanEquals[S <: ScalaType, D2 <: DType](
      t1: S,
      t2: Tensor[D2]
  ): Tensor[Div[D2, ScalaToDType[S]]] =
    fromNative(torchNative.greaterThanEquals(toScalar(t1), t2.native))

  def greaterThan[S <: ScalaType, D2 <: DType](
      t1: S,
      t2: Tensor[D2]
  ): Tensor[Div[D2, ScalaToDType[S]]] =
    fromNative(torchNative.greaterThan(toScalar(t1), t2.native))

  def lessThanEquals[S <: ScalaType, D2 <: DType](
      t1: S,
      t2: Tensor[D2]
  ): Tensor[Div[D2, ScalaToDType[S]]] =
    fromNative(torchNative.lessThanEquals(toScalar(t1), t2.native))

  def lessThan[S <: ScalaType, D2 <: DType](
      t1: S,
      t2: Tensor[D2]
  ): Tensor[Div[D2, ScalaToDType[S]]] =
    fromNative(torchNative.lessThan(toScalar(t1), t2.native))

  def xor[S <: ScalaType, D2 <: DType](t1: S, t2: Tensor[D2]): Tensor[Div[D2, ScalaToDType[S]]] =
    fromNative(torchNative.xor(toScalar(t1), t2.native))

  def or[S <: ScalaType, D2 <: DType](t1: S, t2: Tensor[D2]): Tensor[Div[D2, ScalaToDType[S]]] =
    fromNative(torchNative.or(toScalar(t1), t2.native))

  def and[S <: ScalaType, D2 <: DType](t1: S, t2: Tensor[D2]): Tensor[Div[D2, ScalaToDType[S]]] =
    fromNative(torchNative.and(toScalar(t1), t2.native))

  def mod[S <: ScalaType, D2 <: DType](t1: S, t2: Tensor[D2]): Tensor[Div[D2, ScalaToDType[S]]] =
    fromNative(torchNative.mod(toScalar(t1), t2.native))

  def eq[D1 <: DType, S <: ScalaType](t1: Tensor[D1], other: S): Tensor[Div[D1, ScalaToDType[S]]] =
    fromNative(torchNative.eq(t1.native, toScalar(other)))

  def greaterThanEquals[D1 <: DType, S <: ScalaType](
      t1: Tensor[D1],
      other: S
  ): Tensor[Div[D1, ScalaToDType[S]]] =
    fromNative(torchNative.greaterThanEquals(t1.native, toScalar(other)))

  def greaterThan[D1 <: DType, S <: ScalaType](
      t1: Tensor[D1],
      other: S
  ): Tensor[Div[D1, ScalaToDType[S]]] =
    fromNative(torchNative.greaterThan(t1.native, toScalar(other)))

  def lessThanEquals[D1 <: DType, S <: ScalaType](
      t1: Tensor[D1],
      other: S
  ): Tensor[Div[D1, ScalaToDType[S]]] =
    fromNative(torchNative.lessThanEquals(t1.native, toScalar(other)))

  def lessThan[D1 <: DType, S <: ScalaType](
      t1: Tensor[D1],
      other: S
  ): Tensor[Div[D1, ScalaToDType[S]]] =
    fromNative(torchNative.lessThan(t1.native, toScalar(other)))

  def xor[D1 <: DType, S <: ScalaType](t1: Tensor[D1], other: S): Tensor[Div[D1, ScalaToDType[S]]] =
    fromNative(torchNative.xor(t1.native, toScalar(other)))

  def or[D1 <: DType, S <: ScalaType](t1: Tensor[D1], other: S): Tensor[Div[D1, ScalaToDType[S]]] =
    fromNative(torchNative.or(t1.native, toScalar(other)))

  def and[D1 <: DType, S <: ScalaType](t1: Tensor[D1], other: S): Tensor[Div[D1, ScalaToDType[S]]] =
    fromNative(torchNative.and(t1.native, toScalar(other)))

  def mod[D1 <: DType, S <: ScalaType](t1: Tensor[D1], other: S): Tensor[Div[D1, ScalaToDType[S]]] =
    fromNative(torchNative.mod(t1.native, toScalar(other)))

  def trapz[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.trapz(t1.native, t2.native))

  def trapezoid[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.trapezoid(t1.native, t2.native))

  def take_along_dim_raw[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.take_along_dim(t1.native, t2.native))

  def subtract[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.subtract(t1.native, t2.native))

//  def sub[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.sub(t1.native, t2.native))

//  def square[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.square(t1.native))

//  def std[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.std(t1.native, true))

  def t_copy[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.t_copy(t1.native))

//  def tan[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.tan(t1.native))
//
//  def tanh[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.tanh(t1.native))

//  def trace[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.trace(t1.native))

  def tril_indices[D1 <: DType](
      row: Long,
      col: Long,
      offset: Long,
      dtype: D1,
      device: Device = CPU,
      layout: Layout = Strided,
      requires_grad: Boolean = false
  ): Tensor[D1] =
    val options = NativeConverters.tensorOptions(dtype, layout, device, requires_grad)
    fromNative(torchNative.tril_indices(row, col, offset, options))

  def triu_indices[D1 <: DType](
      row: Long,
      col: Long,
      offset: Long,
      dtype: D1,
      device: Device = CPU,
      layout: Layout = Strided,
      requires_grad: Boolean = false
  ): Tensor[D1] =
    val options = NativeConverters.tensorOptions(dtype, layout, device, requires_grad)
    fromNative(torchNative.triu_indices(row, col, offset, options))

  def tril_indices[D1 <: DType](row: Long, col: Long): Tensor[D1] =
    fromNative(torchNative.tril_indices(row, col))

  def triu_indices[D1 <: DType](row: Long, col: Long): Tensor[D1] =
    fromNative(torchNative.triu_indices(row, col))

  def trapezoid[D1 <: DType](t1: Tensor[D1], t2: Tensor[D1], dim: Int): Tensor[D1] =
    fromNative(torchNative.trapezoid(t1.native, t2.native, dim.toLong))

  // torch.isin(elements, test_elements, *, assume_unique=False, invert=False)
  def isin[D1 <: DType](elements: Tensor[D1], test_elements: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.isin(elements.native, test_elements.native))

  def isin[D1 <: DType](
      elements: Tensor[D1],
      test_elements: Tensor[D1],
      assume_unique: Boolean = false,
      invert: Boolean = false
  ): Tensor[D1] =
    fromNative(torchNative.isin(elements.native, test_elements.native, assume_unique, invert))

  def isFloatingType[D1 <: DType](t1: D1): Boolean =
    torchNative.isFloatingType(t1.toScalarType)

  def isFloat8Type[D1 <: DType](t1: D1): Boolean =
    torchNative.isFloat8Type(t1.toScalarType)

  def isComplexType[D1 <: DType](t1: D1): Boolean =
    torchNative.isComplexType(t1.toScalarType)

  def isIntegralType[D1 <: DType](t1: D1, include_bool: Boolean = false): Boolean =
    torchNative.isIntegralType(t1.toScalarType, include_bool)

  def isReducedFloatingType[D1 <: DType](t1: D1): Boolean =
    torchNative.isReducedFloatingType(t1.toScalarType)

  def isBitsType[D1 <: DType](t1: D1): Boolean =
    torchNative.isBitsType(t1.toScalarType)

  def isBarebonesUnsignedType[D1 <: DType](t1: D1): Boolean = {
    torchNative.isBarebonesUnsignedType(t1.toScalarType)
  }

  def isQIntType[D1 <: DType](t1: D1): Boolean = {
    torchNative.isQIntType(t1.toScalarType)
  }

  //  def trapezoid[D1 <: DType](t1: Tensor[D1], t2: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.trapezoid(t1.native, t2.native))

  def trapezoid[D1 <: DType, S <: ScalaType](t1: Tensor[D1], t2: S, dim: Int): Tensor[D1] =
    fromNative(torchNative.trapezoid(t1.native, toScalar(t2), dim.toLong))

  def trapezoid[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.trapezoid(t1.native))

  def trapz[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.trapz(t1.native))

  def trapz[D1 <: DType](t1: Tensor[D1], xy: Double, dim: Int): Tensor[D1] =
    fromNative(torchNative.trapz(t1.native, xy, dim.toLong))

  def trapz[D1 <: DType](t1: Tensor[D1], t2: Tensor[D1], dim: Int): Tensor[D1] =
    fromNative(torchNative.trapz(t1.native, t2.native, dim.toLong))

  def tril_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.tril(t1.native))

//  def tril[D1 <: DType](t1: Tensor[D1], diagonal: Int = 0): Tensor[D1] =
//    fromNative(torchNative.tril(t1.native, diagonal))

  def triu[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.triu(t1.native))

  def triu[D1 <: DType](t1: Tensor[D1], diagonal: Int = 0): Tensor[D1] =
    fromNative(torchNative.triu(t1.native, diagonal))

//  def trunc[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.trunc(t1.native))

//  def unsqueeze_raw[D1 <: DType](t1: Tensor[D1], dim: Long): Tensor[D1] =
//    fromNative(torchNative.unsqueeze(t1.native, dim))

  def vander[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.vander(t1.native))

  def var_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.`var`(t1.native))

  def view_as_real[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.view_as_real(t1.native))

  def view_as_complex[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.view_as_complex(t1.native))

  def avg_pool3d[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.avg_pool3d(t1.native, sliceSeq*))

  def avg_pool2d[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.avg_pool2d(t1.native, sliceSeq*))

  def avg_pool1d[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.avg_pool1d(t1.native, sliceSeq*))

  def adaptive_avg_pool3d[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.adaptive_avg_pool3d(t1.native, sliceSeq*))

  def adaptive_avg_pool2d[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.adaptive_avg_pool2d(t1.native, sliceSeq*))

  def adaptive_avg_pool1d[D1 <: DType](t1: Tensor[D1], sliceSeq: Seq[Long]): Tensor[D1] =
    fromNative(torchNative.adaptive_avg_pool1d(t1.native, sliceSeq*))

//  def conj[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.conj(t1.native))

  def fft_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.fft(t1.native))

  def ifft_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.ifft(t1.native))

  def fft2_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.fft2(t1.native))

  def ifft2_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.ifft2(t1.native))

  def fftn_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.fftn(t1.native))

  def ifftn_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.ifftn(t1.native))

  def rfft_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.rfft(t1.native))

  def irfft_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.irfft(t1.native))

  def rfft2_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.rfft2(t1.native))

  def irfft2_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.irfft2(t1.native))

  def rfftn_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.rfftn(t1.native))

  def irfftn_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.irfftn(t1.native))

  def hfft_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.hfft(t1.native))

  def ihfft_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.ihfft(t1.native))

  def hfft2_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.hfft2(t1.native))

  def ihfft2_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.ihfft2(t1.native))

  def hfftn_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.hfftn(t1.native))

  def ihfftn_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.ihfftn(t1.native))

  def fftshift_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.fftshift(t1.native))

  def ifftshift_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.ifftshift(t1.native))

  def eigvals[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.linalg_eigvals(t1.native))

  def multi_dot[D1 <: DType](tensorArray: Seq[Tensor[D1]]): Tensor[D1] =
    val tensorVector = TensorVector(tensorArray.map(_.native).toArray*)
    fromNative(torchNative.linalg_multi_dot(tensorVector))

  def pinv[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.linalg_pinv(t1.native))

  def dropout[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.dropout(t1.native))

  def dropout2d[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.dropout2d(t1.native))

  def alpha_dropout[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.alpha_dropout(t1.native))

  def feature_alpha_dropout[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.feature_alpha_dropout(t1.native))

  def instance_norm[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.instance_norm(t1.native))

  def logsigmoid[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.logsigmoid(t1.native))

  def gumbel_softmax[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.gumbel_softmax(t1.native))

  def tanhshrink[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.tanhshrink(t1.native))

  def xavier_normal_[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.xavier_normal_(t1.native))

  def xavier_uniform_[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.xavier_uniform_(t1.native))

  def kaiming_normal_[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.kaiming_normal_(t1.native))

  def kaiming_uniform_[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.kaiming_uniform_(t1.native))

  def uniform_[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.uniform_(t1.native))

  def orthogonal_[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.orthogonal_(t1.native))

//  def zeros_[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.zeros_(t1.native))

  def gammaln[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.gammaln(t1.native))

  def psi[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.psi(t1.native))

  def entr[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.entr(t1.native))

  def ndtri[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.ndtri(t1.native))

  def log_ndtr[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.log_ndtr(t1.native))

  def expit[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.expit(t1.native))

//  def argwhere[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.argwhere(t1.native))

//  def argmin[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.argmin(t1.native))
//
//  def argmax[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.argmax(t1.native))

  def arctanh[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.arctanh(t1.native))

  def bincount[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.bincount(t1.native))

  def bernoulli_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.bernoulli(t1.native))

  def atleast_3d[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.atleast_3d(t1.native))

  def atleast_2d[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.atleast_2d(t1.native))

  def atleast_1d[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.atleast_1d(t1.native))

//  def atanh[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.atanh(t1.native))

//  def atan[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.atan(t1.native))

//  def asinh[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.asinh(t1.native))
//
//  def asin[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.asin(t1.native))

//  def argsort[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.argsort(t1.native))

  def arctan[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.arctan(t1.native))

  def arcsinh[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.arcsinh(t1.native))

  def arcsin[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.arcsin(t1.native))

  def arccosh[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.arccosh(t1.native))

  def arccos[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.arccos(t1.native))

//  def any[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.any(t1.native))

//  def angle[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.angle(t1.native))
//

//  def amin[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.amin(t1.native))
//
//  def amax[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.amax(t1.native))

//  def all[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.all(t1.native))

  def alias[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.alias(t1.native))

//  def adjoint[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.adjoint(t1.native))

//  def acosh[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.acosh(t1.native))

//  def acos[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.acos(t1.native))

  def absolute[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.absolute(t1.native))

  def flatten[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.flatten(t1.native))

  def flatten[D1 <: DType](t1: Tensor[D1], start_dim: Int = 1, end_dim: Int = -1): Tensor[D1] =
    fromNative(torchNative.flatten(t1.native, start_dim.toLong, end_dim.toLong))

  import org.bytedeco.pytorch.{FoldOptions, UnfoldOptions}

  def fold[D1 <: DType](
      t1: Tensor[D1],
      output_size: Int | (Int, Int) | (Int, Int, Int),
      kernel_size: Int | (Int, Int) | (Int, Int, Int),
      dilation: Int | (Int, Int) | Option[Int] | Option[(Int, Int)] = 1,
      padding: Int | (Int, Int) | Option[Int] | Option[(Int, Int)] = 0,
      stride: Int | (Int, Int) = 1
  ): Tensor[D1] = {

    val options = FoldOptions(toNative(output_size), toNative(kernel_size))

    dilation match {
      case d: Int                => options.dilation().put(toNative(d))
      case d: (Int, Int)         => options.dilation().put(toNative(d))
      case d: Option[Int]        => if d.isDefined then options.dilation().put(toNative(d.get))
      case d: Option[(Int, Int)] => if d.isDefined then options.dilation().put(toNative(d.get))
    }
    padding match {
      case p: Int                => options.padding().put(toNative(p))
      case p: (Int, Int)         => options.padding().put(toNative(p))
      case p: Option[Int]        => if p.isDefined then options.padding().put(toNative(p.get))
      case p: Option[(Int, Int)] => if p.isDefined then options.padding().put(toNative(p.get))
    }

    stride match {
      case s: Int        => options.stride().put(Array(s.toLong)*)
      case s: (Int, Int) => options.stride().put(Array(s._1.toLong, s._2.toLong)*)
    }

    output_size match {
      case s: Int        => options.output_size().put(Array(s.toLong)*)
      case s: (Int, Int) => options.output_size().put(Array(s._1.toLong, s._2.toLong)*)
      case s: (Int, Int, Int) =>
        options.output_size().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
    }

    kernel_size match {
      case s: Int        => options.kernel_size().put(Array(s.toLong)*)
      case s: (Int, Int) => options.kernel_size().put(Array(s._1.toLong, s._2.toLong)*)
      case s: (Int, Int, Int) =>
        options.kernel_size().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
    }
    fromNative(torchNative.fold(t1.native, options))

  }

  def unfold[D1 <: DType](
      t1: Tensor[D1],
      kernel_size: Int | (Int, Int) | (Int, Int, Int),
      dilation: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
        Option[(Int, Int, Int)] = 1,
      padding: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
        Option[(Int, Int, Int)] = 0,
      stride: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
        Option[(Int, Int, Int)] = 1
  ): Tensor[D1] = {
    val options = UnfoldOptions(toNative(kernel_size))
    options.kernel_size().put(toNative(kernel_size))
    stride match {
      case s: Int                     => options.stride().put(toNative(s))
      case s: (Int, Int)              => options.stride().put(toNative(s))
      case s: (Int, Int, Int)         => options.stride().put(toNative(s))
      case s: Option[Int]             => if s.isDefined then options.stride().put(toNative(s.get))
      case s: Option[(Int, Int)]      => if s.isDefined then options.stride().put(toNative(s.get))
      case s: Option[(Int, Int, Int)] => if s.isDefined then options.stride().put(toNative(s.get))
    }
    padding match {
      case s: Int                     => options.padding().put(toNative(s))
      case s: (Int, Int)              => options.padding().put(toNative(s))
      case s: (Int, Int, Int)         => options.padding().put(toNative(s))
      case s: Option[Int]             => if s.isDefined then options.padding().put(toNative(s.get))
      case s: Option[(Int, Int)]      => if s.isDefined then options.padding().put(toNative(s.get))
      case s: Option[(Int, Int, Int)] => if s.isDefined then options.padding().put(toNative(s.get))
    }
    dilation match {
      case s: Int                     => options.dilation().put(toNative(s))
      case s: (Int, Int)              => options.dilation().put(toNative(s))
      case s: (Int, Int, Int)         => options.dilation().put(toNative(s))
      case s: Option[Int]             => if s.isDefined then options.dilation().put(toNative(s.get))
      case s: Option[(Int, Int)]      => if s.isDefined then options.dilation().put(toNative(s.get))
      case s: Option[(Int, Int, Int)] => if s.isDefined then options.dilation().put(toNative(s.get))
    }
    fromNative(torchNative.unfold(t1.native, options))
  }

  def unflatten[D1 <: DType](t1: Tensor[D1], dim: Int = 1, unflattened_size: Seq[Int]): Tensor[D1] =
    fromNative(
      torchNative.unflatten(t1.native, dim.toLong, unflattened_size.map(_.toLong).toArray*)
    )

  def matrix_power[D1 <: DType](t1: Tensor[D1], num: Int): Tensor[D1] =
    fromNative(torchNative.matrix_power(t1.native, num))

  def empty_like_raw[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.empty_like(t1.native))

//  def empty[D1 <: DType](t1: Seq[Long]): Tensor[D1] =
//    fromNative(torchNative.empty(t1 *))

  def kthvalue[D1 <: DType](t1: Tensor[D1], sliceSeq: Long): (Tensor[D1], Tensor[D1]) =
    val tup = torchNative.kthvalue(t1.native, sliceSeq)
    (fromNative(tup.get0), fromNative(tup.get1))

  def linalg_qr[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1]) =
    val tup = torchNative.linalg_qr(t1.native)
    (fromNative(tup.get0), fromNative(tup.get1))

  def linalg_slogdet[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1]) =
    val tup = torchNative.linalg_slogdet(t1.native)
    (fromNative(tup.get0), fromNative(tup.get1))

  def linalg_svd[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1], Tensor[D1]) =
    val tup = torchNative.linalg_svd(t1.native)
    (fromNative(tup.get0), fromNative(tup.get1), fromNative(tup.get2))

//  def mode[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1]) =
//    val tup = torchNative.mode(t1.native)
//    (fromNative(tup.get0), fromNative(tup.get1))

  def flattenDenseTensors[D1 <: DType](tensorArray: Seq[Tensor[D1]]): Tensor[D1] =
    val tensorVector = TensorVector(tensorArray.map(_.native).toArray*)
    fromNative(torchNative.flattenDenseTensors(tensorVector))

  def flatten_dense_tensors[D1 <: DType](tensorArray: Seq[Tensor[D1]]): Tensor[D1] =
    val tensorVector = TensorVector(tensorArray.map(_.native).toArray*)
    fromNative(torchNative.flatten_dense_tensors(tensorVector))

  def unflatten_dense_tensors[D1 <: DType](
      tensor: Tensor[D1],
      tensorArray: Seq[Tensor[D1]]
  ): Seq[Tensor[D1]] =
    val tensorVector = TensorVector(tensorArray.map(_.native).toArray*)
    val denseTensorSeq = torchNative.unflatten_dense_tensors(tensor.native, tensorVector)
    tensorVectorToSeqTensor(denseTensorSeq)

  def newLikeFlat[D1 <: DType](tensorArray: Seq[Tensor[D1]]): Tensor[D1] =
    val tensorVector = TensorVector(tensorArray.map(_.native).toArray*)
    fromNative(torchNative.newLikeFlat(tensorVector))

  def bilinear[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      t3: Tensor[D1 | D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.bilinear(t1.native, t2.native, t3.native))

  def baddbmm[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      t3: Tensor[D1 | D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.baddbmm(t1.native, t2.native, t3.native))

  def baddbmm[D1 <: DType, D2 <: DType, S <: ScalaType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      t3: Tensor[D1 | D2],
      beta: S,
      alpha: S
  ): Tensor[Promoted[D1, D2]] =
    fromNative(
      torchNative.baddbmm(t1.native, t2.native, t3.native, toScalar(beta), toScalar(alpha))
    )

  def addr[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      t3: Tensor[D1 | D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.addr(t1.native, t2.native, t3.native))

  def addmv[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      t3: Tensor[D1 | D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.addmv(t1.native, t2.native, t3.native))

  def addmm[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      t3: Tensor[D1 | D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.addmm(t1.native, t2.native, t3.native))

//  def addcmul_raw[D1 <: DType, D2 <: DType](
//      t1: Tensor[D1],
//      t2: Tensor[D2],
//      t3: Tensor[D1 | D2]
//  ): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.addcmul(t1.native, t2.native, t3.native))
//
//  def addcdiv_raw[D1 <: DType, D2 <: DType](
//      t1: Tensor[D1],
//      t2: Tensor[D2],
//      t3: Tensor[D1 | D2]
//  ): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.addcdiv(t1.native, t2.native, t3.native))

  def addbmm[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      t3: Tensor[D1 | D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.addbmm(t1.native, t2.native, t3.native))

  def slogdet[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1]) =
    val tup = torchNative.slogdet(t1.native)
    (fromNative(tup.get0), fromNative(tup.get1))

//  def atan2[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.atan2(t1.native, t2.native))

//  def polygamma[D1 <: DType](t1: Tensor[D1], lon: Long): Tensor[D1] =
//    fromNative(torchNative.polygamma(lon, t1.native))

  def linalg_solve_ex[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): (Tensor[Promoted[D1, D2]], Tensor[Promoted[D1, D2]]) =
    val tup = torchNative.linalg_solve_ex(t1.native, t2.native)
    (fromNative(tup.get0), fromNative(tup.get1))

  def linalg_solve_triangular[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      bool: Boolean
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.linalg_solve_triangular(t1.native, t2.native, bool))

  def lu_unpack[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): (Tensor[Promoted[D1, D2]], Tensor[Promoted[D1, D2]], Tensor[Promoted[D1, D2]]) =
    val tup = torchNative.lu_unpack(t1.native, t2.native)
    (fromNative(tup.get0), fromNative(tup.get1), fromNative(tup.get2))

  def scaled_dot_product_attention_raw[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      t3: Tensor[D1 | D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.scaled_dot_product_attention(t1.native, t2.native, t3.native))

  def sparse_sampled_addmm[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      t3: Tensor[D1 | D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.sparse_sampled_addmm(t1.native, t2.native, t3.native))

  def triplet_margin_loss[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      t3: Tensor[D1 | D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.triplet_margin_loss(t1.native, t2.native, t3.native))

//  def where[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2], t3: Tensor[D1 | D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.where(t1.native, t2.native, t3.native))

  def batch_norm[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      t3: Tensor[D1 | D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.batch_norm(t1.native, t2.native, t3.native))

  def std_mean_raw[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1]) =
    val tup = torchNative.std_mean(t1.native)
    (fromNative(tup.get0), fromNative(tup.get1))

  def svd[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1], Tensor[D1]) =
    val triple = torchNative.svd(t1.native)
    (fromNative(triple.get0), fromNative(triple.get1), fromNative(triple.get2))

//  def take[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.take(t1.native, t2.native))

  def var_mean_raw[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1]) =
    val tup = torchNative.var_mean(t1.native)
    (fromNative(tup.get0), fromNative(tup.get1))

  def ctc_loss[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      t3: Tensor[D1 | D2],
      t4: Tensor[D1 | D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.ctc_loss(t1.native, t2.native, t3.native, t4.native))

  def lu_factor[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1]) =
    val tup = torchNative.linalg_lu_factor(t1.native)
    (fromNative(tup.get0), fromNative(tup.get1))

  def sparse_[D1 <: DType](t1: Tensor[D1], dou: Double): Tensor[D1] =
    fromNative(torchNative.sparse_(t1.native, dou))

  def eig[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1]) =
    val tup = torchNative.linalg_eig(t1.native)
    (fromNative(tup.get0), fromNative(tup.get1))

  def eigh[D1 <: DType](t1: Tensor[D1], str: String): (Tensor[D1], Tensor[D1]) =
    val tup = torchNative.linalg_eigh(t1.native, str)
    (fromNative(tup.get0), fromNative(tup.get1))

  def sspaddmm[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      t3: Tensor[D1 | D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.sspaddmm(t1.native, t2.native, t3.native))

  def triangular_solve[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): (Tensor[Promoted[D1, D2]], Tensor[Promoted[D1, D2]]) =
    val tup = torchNative.triangular_solve(t1.native, t2.native)
    (fromNative(tup.get0), fromNative(tup.get1))

  def unique_consecutive[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1], Tensor[D1]) =
    val triple = torchNative.unique_consecutive(t1.native)
    (fromNative(triple.get0), fromNative(triple.get1), fromNative(triple.get2))

  def lu[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1], Tensor[D1]) =
    val tup = torchNative.linalg_lu(t1.native)
    (fromNative(tup.get0), fromNative(tup.get1), fromNative(tup.get2))

  def aminmax_raw[D1 <: DType](t1: Tensor[D1]): (Tensor[D1], Tensor[D1]) =
    val tup = torchNative.aminmax(t1.native)
    (fromNative(tup.get0), fromNative(tup.get1))

}
