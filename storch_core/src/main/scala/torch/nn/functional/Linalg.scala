package torch
package nn
package functional

import org.bytedeco.pytorch.{DoubleOptional, LongArrayRefOptional}
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.fromNative

//@Namespace("torch::linalg
private[torch] trait Linalg {

  def cholesky_inverse[D <: DType](input: Tensor[D]): Tensor[D] = {
    val result = torchNative.cholesky_inverse(input.native)
    fromNative(result)
  }

  def cholesky_solve[D <: DType](input: Tensor[D], input2: Tensor[D]): Tensor[D] = {
    val result = torchNative.cholesky_solve(input.native, input2.native)
    fromNative(result)
  }

  def pinverse[D <: DType](input: Tensor[D], rcond: Double = Double.PositiveInfinity): Tensor[D] = {
    val result = torchNative.pinverse(input.native, rcond)
    fromNative(result)
  }

//  def pstrf[D <: DType](input: Tensor[D], upper: Boolean = false, transpose: Boolean = false, driver: String = "gesvd"): Tensor[D] = {
//    val result = torchNative.pstrf(input.native, upper, transpose, driver)
//    fromNative(result)
//  }

  def qr[D <: DType](input: Tensor[D], some: Boolean = false): Tuple2[Tensor[D], Tensor[D]] = {
    val result = torchNative.qr(input.native, some)
    val t1 = fromNative[D](result.get0())
    val t2 = fromNative[D](result.get1())
    (t1, t2)
  }

  def svd[D <: DType](
      input: Tensor[D],
      some: Boolean = false,
      computeU: Boolean = true,
      computeV: Boolean = true
  ): Tuple3[Tensor[D], Tensor[D], Tensor[D]] = {
    val result = torchNative.svd(input.native, computeU, computeV)
    val t1 = fromNative[D](result.get0())
    val t2 = fromNative[D](result.get1())
    val t3 = fromNative[D](result.get2())
    (t1, t2, t3)
  }

  def eig[D <: DType](input: Tensor[D], some: Boolean = false): Tuple2[Tensor[D], Tensor[D]] = {
    val result = torchNative.linalg_eig(input.native)
    val t1 = fromNative[D](result.get0())
    val t2 = fromNative[D](result.get1())
    (t1, t2)
  }

  def eigvals[D <: DType](input: Tensor[D], some: Boolean = false): Tensor[D] = {
    val result = torchNative.linalg_eigvals(input.native)
    fromNative(result)
  }

  def eigh[D <: DType](
      input: Tensor[D],
      some: Boolean = false,
      upper: Boolean = false
  ): Tuple2[Tensor[D], Tensor[D]] = {
    val result = torchNative.linalg_eigh(input.native, upper.toString)
    val t1 = fromNative[D](result.get0())
    val t2 = fromNative[D](result.get1())
    (t1, t2)
  }

//  def eighvals[D <: DType](input: Tensor[D], some: Boolean = false, upper: Boolean = false): Tensor[D] = {
//    val result = torchNative.eighvals(input.native, some, upper)
//    fromNative(result)
//  }

  def eigvalsh[D <: DType](
      input: Tensor[D],
      some: Boolean = false,
      upper: Boolean = false
  ): Tensor[D] = {
    val result = torchNative.linalg_eigvalsh(input.native, upper.toString)
    fromNative(result)
  }

  def lu[D <: DType](
      input: Tensor[D],
      some: Boolean = false
  ): Tuple3[Tensor[D], Tensor[D], Tensor[D]] = {

    val result = torchNative.linalg_lu(input.native, some)
    val t1 = fromNative[D](result.get0())
    val t2 = fromNative[D](result.get1())
    val t3 = fromNative[D](result.get2())
    (t1, t2, t3)
  }

//  @Namespace("torch::linalg")
//  @ByVal
//  @native def vector_norm(@Const @ByRef var0: Tensor, @ByVal var1: Scalar,
//                          @ByVal var2: LongArrayRefOptional,
//                          @Cast(Array(Array("bool"))) var3: Boolean,
//                          @ByVal var4: ScalarTypeOptional): Tensor
//
//  @Namespace("torch::linalg")
//  @ByVal
//  @native def vector_norm(@Const @ByRef var0: Tensor, @ByVal var1: Scalar,
//                          @ByVal @Cast(Array(Array("int64_t*", "c10::ArrayRef<int64_t>", "std::vector<int64_t>&"))) @StdVector var2: Array[Long],
//                          @Cast(Array(Array("bool"))) var3: Boolean,
//                          @ByVal var4: ScalarTypeOptional): Tensor

//  def vector_norm[D <: DType](input: Tensor[D], p: Double = 2.0, keepdim: Boolean = false): Tensor[D] = {
//    val result = torchNative.vector_norm(input.native, p, keepdim)
//    fromNative(result)
//  }
//
//  def matrix_norm[D <: DType](input: Tensor[D], p: Double = 2.0, keepdim: Boolean = false): Tensor[D] = {
//    val result = torchNative.matrix_norm(input.native, p, keepdim)
//    fromNative(result)
//  }
//
//  def diagonal[D <: DType](input: Tensor[D], offset: Int = 0, diagonal: Int = 0, wrap: Boolean = false): Tensor[D] = {
//    val result = torchNative.diagonal(input.native, offset, diagonal, wrap)
//    fromNative(result)
//  }

  def det[D <: DType](input: Tensor[D]): Tensor[D] = {
    val result = torchNative.det(input.native)
    fromNative(result)
  }

  def slogdet[D <: DType](input: Tensor[D]): Tuple2[Tensor[D], Tensor[D]] = {
    val result = torchNative.slogdet(input.native)
    val t1 = fromNative[D](result.get0())
    val t2 = fromNative[D](result.get1())
    (t1, t2)
  }

  def matrix_rank[D <: DType](
      input: Tensor[D],
      tol: Double = Double.PositiveInfinity,
      asum: Double = Double.PositiveInfinity,
      hermitian: Boolean
  ): Tensor[D] = {
    val result =
      torchNative.linalg_matrix_rank(
        input.native,
        DoubleOptional(tol),
        DoubleOptional(asum),
        hermitian
      )
    fromNative(result)
  }

  def lu_factor[D <: DType](input: Tensor[D]): Tuple2[Tensor[D], Tensor[D]] = {
    val result = torchNative.linalg_lu_factor(input.native)
    val t1 = fromNative[D](result.get0())
    val t2 = fromNative[D](result.get1())
    (t1, t2)
  }

  def lu_solve[D <: DType](input: Tensor[D], input2: Tensor[D], input3: Tensor[D]): Tensor[D] = {
    val result = torchNative.lu_solve(input.native, input2.native, input3.native)
    fromNative(result)
  }

  def tensorinv[D <: DType](input: Tensor[D], ind: Int): Tensor[D] = {
    val result = torchNative.linalg_tensorinv(input.native, ind)
    fromNative(result)
  }

  def tensorsolve[D <: DType](input: Tensor[D], input2: Tensor[D], var2: Seq[Long]): Tensor[D] = {
    val ref = LongArrayRefOptional(var2: _*)
    val result = torchNative.linalg_tensorsolve(input.native, input2.native, ref)
    fromNative(result)
  }

  def cholesky_ex[D <: DType](
      input: Tensor[D],
      upper: Boolean = false,
      var2: Boolean = false
  ): Tuple2[Tensor[D], Tensor[D]] = {
    val result = torchNative.linalg_cholesky_ex(input.native, upper, var2)
    val t1 = fromNative[D](result.get0())
    val t2 = fromNative[D](result.get1())
    (t1, t2)
  }

  def inv_ex[D <: DType](input: Tensor[D], upper: Boolean = false): Tuple2[Tensor[D], Tensor[D]] = {
    val result = torchNative.linalg_inv_ex(input.native, upper)
    val t1 = fromNative[D](result.get0())
    val t2 = fromNative[D](result.get1())
    (t1, t2)
  }

  def solve_ex[D <: DType](
      input: Tensor[D],
      input2: Tensor[D],
      upper: Boolean = false,
      var3: Boolean = false
  ): Tuple2[Tensor[D], Tensor[D]] = {
    val result = torchNative.linalg_solve_ex(input.native, input2.native, upper, var3)
    val t1 = fromNative[D](result.get0())
    val t2 = fromNative[D](result.get1())
    (t1, t2)
  }

  def lu_factor_ex[D <: DType](
      input: Tensor[D],
      var1: Boolean
  ): Tuple3[Tensor[D], Tensor[D], Tensor[D]] = {
    val result = torchNative.linalg_lu_factor_ex(input.native)
    val t1 = fromNative[D](result.get0())
    val t2 = fromNative[D](result.get1())
    val t3 = fromNative[D](result.get2())
    (t1, t2, t3)
  }

  def linalg_ldl_factor[D <: DType](
      input: Tensor[D],
      var1: Boolean
  ): Tuple2[Tensor[D], Tensor[D]] = {
    val result = torchNative.linalg_ldl_factor(input.native, var1)
    val t1 = fromNative[D](result.get0())
    val t2 = fromNative[D](result.get1())
    (t1, t2)
  }

  def ldl_factor_ex[D <: DType](
      input: Tensor[D],
      var1: Boolean,
      var2: Boolean
  ): Tuple2[Tensor[D], Tensor[D]] = {
    val result = torchNative.linalg_ldl_factor_ex(input.native, var1, var2)
    val t1 = fromNative[D](result.get0())
    val t2 = fromNative[D](result.get1())
    (t1, t2)
  }

  def ldl_solve[D <: DType](
      input: Tensor[D],
      input2: Tensor[D],
      input3: Tensor[D],
      var3: Boolean
  ): Tensor[D] = {
    val result = torchNative.linalg_ldl_solve(input.native, input2.native, input3.native, var3)
    fromNative(result)
  }

  def matrix_power[D <: DType](input: Tensor[D], n: Long): Tensor[D] = {
    val result = torchNative.matrix_power(input.native, n)
    fromNative(result)
  }

  def matrix_exp[D <: DType](input: Tensor[D]): Tensor[D] = {
    val result = torchNative.matrix_exp(input.native)
    fromNative(result)
  }

  //  def matrix_log[D <: DType](input:Tensor[D]):Tensor[D] = {
  //    val result = torchNative.matrix_log(input.native)
  //    fromNative(result)
  //  }

}
