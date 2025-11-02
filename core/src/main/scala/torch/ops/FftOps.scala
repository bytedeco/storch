package torch
package ops

import org.bytedeco.pytorch.{
  BoolOptional,
  LongArrayRefOptional,
  LongOptional,
  StringViewOptional,
  SymInt,
  SymIntOptional,
  TensorOptional,
  TensorOptions,
  TensorVector
}
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.fromNative
import org.bytedeco.javacpp.annotation.{ByRef, ByVal, Const, Namespace}
import Layout.Strided
import Device.CPU
import internal.NativeConverters
import NativeConverters.*
import torch.Layout.{Sparse, SparseBsc, SparseBsr, SparseCsc, SparseCsr, Strided}
import torch.numpy.matrix.NDArray
import scala.reflect.ClassTag

trait FftOps {

  def hamming_window[D <: DType](window_length: Int): Tensor[D] = fromNative(
    torchNative.hamming_window(window_length.toLong)
  )

  // torch.hann_window(window_length, periodic=True, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
  def hann_window[D <: DType](window_length: Int): Tensor[D] = fromNative(
    torchNative.hann_window(window_length.toLong)
  )

  def kaiser_window[D <: DType](window_length: Int): Tensor[D] = fromNative(
    torchNative.kaiser_window(window_length.toLong)
  )

  def bartlett_window[D <: DType](window_length: Int): Tensor[D] = fromNative(
    torchNative.bartlett_window(window_length.toLong)
  )

  def blackman_window[D <: DType](window_length: Int): Tensor[D] = fromNative(
    torchNative.blackman_window(window_length.toLong)
  )

  def hamming_window[D <: DType](window_length: Int, periodic: Boolean): Tensor[D] = fromNative(
    torchNative.hamming_window(window_length.toLong, periodic)
  )

  def hamming_window[D <: DType](
      window_length: Int,
      periodic: Boolean,
      alpha: Double,
      beta: Double
  ): Tensor[D] = fromNative(torchNative.hamming_window(window_length.toLong, periodic, alpha, beta))

  def hamming_window[D <: DType](
      window_length: Int,
      periodic: Boolean,
      alpha: Double = 0.54d,
      beta: Double = 0.46d,
      dtype: D,
      device: Device = CPU,
      layout: Layout = Strided,
      requires_grad: Boolean = false
  ): Tensor[D] =
    val options = NativeConverters.tensorOptions(dtype, layout, device, requires_grad)
    fromNative(
      torchNative.hamming_window(window_length.toLong, periodic, alpha, beta, options)
    )

  // torch.hann_window(window_length, periodic=True, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
  def hann_window[D <: DType](window_length: Int, periodic: Boolean): Tensor[D] = fromNative(
    torchNative.hann_window(window_length.toLong, periodic)
  )

  def hann_window[D <: DType](
      window_length: Int,
      periodic: Boolean,
      dtype: D,
      device: Device = CPU,
      layout: Layout = Strided,
      requires_grad: Boolean = false
  ): Tensor[D] =
    val options = NativeConverters.tensorOptions(dtype, layout, device, requires_grad)
    fromNative(torchNative.hann_window(window_length.toLong, periodic, options))

  def kaiser_window[D <: DType](window_length: Int, periodic: Boolean): Tensor[D] = fromNative(
    torchNative.kaiser_window(window_length.toLong, periodic)
  )

  def blackman_window[D <: DType](window_length: Int, periodic: Boolean): Tensor[D] = fromNative(
    torchNative.blackman_window(window_length.toLong, periodic)
  )

  def blackman_window[D <: DType](
      window_length: Int,
      periodic: Boolean,
      dtype: D,
      device: Device = CPU,
      layout: Layout = Strided,
      requires_grad: Boolean = false
  ): Tensor[D] =
    val options = NativeConverters.tensorOptions(dtype, layout, device, requires_grad)
    fromNative(torchNative.blackman_window(window_length.toLong, periodic, options))

  def kaiser_window[D <: DType](
      window_length: Int,
      periodic: Boolean,
      beta: Double
  ): Tensor[D] = fromNative(torchNative.kaiser_window(window_length.toLong, periodic, beta))

  def kaiser_window[D <: DType](
      window_length: Int,
      periodic: Boolean,
      beta: Double = 12.0d,
      dtype: D,
      device: Device = CPU,
      layout: Layout = Strided,
      requires_grad: Boolean = false
  ): Tensor[D] =
    val options = NativeConverters.tensorOptions(dtype, layout, device, requires_grad)
    fromNative(torchNative.kaiser_window(window_length.toLong, periodic, beta, options))

  def bartlett_window[D <: DType](window_length: Int, periodic: Boolean): Tensor[D] = fromNative(
    torchNative.bartlett_window(window_length.toLong, periodic)
  )

  def bartlett_window[D <: DType](
      window_length: Int,
      periodic: Boolean,
      dtype: D,
      device: Device = CPU,
      layout: Layout = Strided,
      requires_grad: Boolean = false
  ): Tensor[D] =
    val options = NativeConverters.tensorOptions(dtype, layout, device, requires_grad)
    fromNative(torchNative.bartlett_window(window_length.toLong, periodic, options))

  def istft[D <: DType](input: Tensor[D], n_fft: Long): Tensor[D] = fromNative(
    torchNative.istft(input.native, n_fft)
  )

  def istft[D <: DType](
      input: Tensor[D],
      n_fft: Long,
      hop_length: Option[Long] = None,
      win_length: Option[Long] = None,
      window: Option[Tensor[D]] = None,
      center: Boolean = true,
      normalized: Boolean = false,
      onesided: Option[Boolean] = None,
      length: Option[Long] = None,
      return_complex: Boolean = false
  ): Tensor[D] =
    val nativeHopLength =
      if (hop_length.isDefined) new LongOptional(hop_length.get.toLong) else new LongOptional()
    val nativeWinLength =
      if (win_length.isDefined) new LongOptional(win_length.get.toLong) else new LongOptional()
    val nativeWindow =
      if (window.isDefined) new TensorOptional(window.get.native) else new TensorOptional()
    val nativeOnesided =
      if (onesided.isDefined) new BoolOptional(onesided.get) else new BoolOptional()
    val nativeLength =
      if (length.isDefined) new LongOptional(length.get.toLong) else new LongOptional()
    fromNative(
      torchNative.istft(
        input.native,
        n_fft,
        nativeHopLength,
        nativeWinLength,
        nativeWindow,
        center,
        normalized,
        nativeOnesided,
        nativeLength,
        return_complex
      )
    )

  def stft[D <: DType](
      input: Tensor[D],
      n_fft: Long,
      hop_length: Option[Long] = None,
      win_length: Option[Long] = None,
      window: Option[Tensor[D]] = None,
      center: Boolean = true,
      pad_mode: String = "reflect",
      normalized: Boolean = true,
      onesided: Option[Boolean] = None,
      return_complex: Option[Boolean] = None,
      align_to_window: Option[Boolean] = None
  ): Tensor[D] = {

    val hl = if hop_length.isDefined then { new LongOptional(hop_length.get) }
    else new LongOptional()
    val wl = if win_length.isDefined then new LongOptional(win_length.get) else new LongOptional()
    val win =
      if window.isDefined then new TensorOptional(window.get.native) else new TensorOptional()
    val os = if onesided.isDefined then new BoolOptional(onesided.get) else new BoolOptional()
    val rc =
      if return_complex.isDefined then new BoolOptional(return_complex.get) else new BoolOptional()
    val aw =
      if align_to_window.isDefined then new BoolOptional(align_to_window.get)
      else new BoolOptional()

    val nt =
      torchNative.stft(input.native, n_fft, hl, wl, win, center, pad_mode, normalized, os, rc, aw)
    fromNative(nt)
  }

  def fft[D <: DType](
      input: Tensor[D],
      n: Option[SymInt] = None,
      dim: Long = -1L,
      norm: Option[String] = None
  ): Tensor[D] = {

    val center = if n.isDefined then new SymIntOptional(n.get) else new SymIntOptional()
    val normStr =
      if norm.isDefined then new StringViewOptional(norm.get) else new StringViewOptional()
    fromNative(torchNative.fft(input.native, center, dim, normStr))
  }

  def ifft[D <: DType](
      input: Tensor[D],
      n: Option[SymInt] = None,
      dim: Long = -1L,
      norm: Option[String] = None
  ): Tensor[D] = {
    val center = if n.isDefined then new SymIntOptional(n.get) else new SymIntOptional()
    val normStr =
      if norm.isDefined then new StringViewOptional(norm.get) else new StringViewOptional()
    fromNative(torchNative.ifft(input.native, center, dim, normStr))
  }

  def fft2[D <: DType](
      input: Tensor[D],
      s: Option[Array[Long]] = None,
      dim: Array[Long] = Array(-2L, -1L),
      norm: Option[String] = None
  ): Tensor[D] = {

    val so = if s.isDefined then s.get else new LongArrayRefOptional()
    val normStr =
      if norm.isDefined then new StringViewOptional(norm.get) else new StringViewOptional()
    val nt =
      if s.isDefined then torchNative.fft2(input.native, s.get, dim, normStr)
      else torchNative.fft2(input.native, new LongArrayRefOptional(), dim, normStr)
    fromNative(nt)
  }
  def ifft2[D <: DType](
      input: Tensor[D],
      s: Option[Array[Long]] = None,
      dim: Array[Long] = Array(-2L, -1L),
      norm: Option[String] = None
  ): Tensor[D] = {

    val so = if s.isDefined then s.get else new LongArrayRefOptional()
    val normStr =
      if norm.isDefined then new StringViewOptional(norm.get) else new StringViewOptional()
    val nt =
      if s.isDefined then torchNative.ifft2(input.native, s.get, dim, normStr)
      else torchNative.ifft2(input.native, new LongArrayRefOptional(), dim, normStr)
    fromNative(nt)
  }
  def fftn[D <: DType](
      input: Tensor[D],
      s: Option[Array[Long]] = None,
      dim: Array[Long] = Array(-2L, -1L),
      norm: Option[String] = None
  ): Tensor[D] = {

    val so = if s.isDefined then s.get else new LongArrayRefOptional()
    val dimOp = new LongArrayRefOptional(dim*)
    val normStr =
      if norm.isDefined then new StringViewOptional(norm.get) else new StringViewOptional()
    val nt =
      if s.isDefined then torchNative.fftn(input.native, s.get, dim, normStr)
      else torchNative.fftn(input.native, new LongArrayRefOptional(), dimOp, normStr)
    fromNative(nt)
  }

  def ifftn[D <: DType](
      input: Tensor[D],
      s: Option[Array[Long]] = None,
      dim: Array[Long] = Array(-2L, -1L),
      norm: Option[String] = None
  ): Tensor[D] = {

    val so = if s.isDefined then s.get else new LongArrayRefOptional()
    val dimOp = new LongArrayRefOptional(dim*)
    val normStr =
      if norm.isDefined then new StringViewOptional(norm.get) else new StringViewOptional()
    val nt =
      if s.isDefined then torchNative.ifftn(input.native, s.get, dim, normStr)
      else torchNative.ifftn(input.native, new LongArrayRefOptional(), dimOp, normStr)
    fromNative(nt)
  }

  def rfft[D <: DType](
      input: Tensor[D],
      n: Option[SymInt] = None,
      dim: Long = -1L,
      norm: Option[String] = None
  ): Tensor[D] = {
    val center = if n.isDefined then new SymIntOptional(n.get) else new SymIntOptional()
    val normStr =
      if norm.isDefined then new StringViewOptional(norm.get) else new StringViewOptional()
    fromNative(torchNative.rfft(input.native, center, dim, normStr))
  }

  def irfft[D <: DType](
      input: Tensor[D],
      n: Option[SymInt] = None,
      dim: Long = -1L,
      norm: Option[String] = None
  ): Tensor[D] = {

    val center = if n.isDefined then new SymIntOptional(n.get) else new SymIntOptional()
    val normStr =
      if norm.isDefined then new StringViewOptional(norm.get) else new StringViewOptional()
    fromNative(torchNative.irfft(input.native, center, dim, normStr))
  }

  def rfft2[D <: DType](
      input: Tensor[D],
      s: Option[Array[Long]] = None,
      dim: Array[Long] = Array(-2L, -1L),
      norm: Option[String] = None
  ): Tensor[D] = {

    val so = if s.isDefined then s.get else new LongArrayRefOptional()
    val normStr =
      if norm.isDefined then new StringViewOptional(norm.get) else new StringViewOptional()
    val nt =
      if s.isDefined then torchNative.rfft2(input.native, s.get, dim, normStr)
      else torchNative.rfft2(input.native, new LongArrayRefOptional(), dim, normStr)
    fromNative(nt)
  }

  def irfft2[D <: DType](
      input: Tensor[D],
      s: Option[Array[Long]] = None,
      dim: Array[Long] = Array(-2L, -1L),
      norm: Option[String] = None
  ): Tensor[D] = {

    val so = if s.isDefined then s.get else new LongArrayRefOptional()
    val normStr =
      if norm.isDefined then new StringViewOptional(norm.get) else new StringViewOptional()
    val nt =
      if s.isDefined then torchNative.irfft2(input.native, s.get, dim, normStr)
      else torchNative.irfft2(input.native, new LongArrayRefOptional(), dim, normStr)
    fromNative(nt)
  }

  def rfftn[D <: DType](
      input: Tensor[D],
      s: Option[Array[Long]] = None,
      dim: Array[Long] = Array(-2L, -1L),
      norm: Option[String] = None
  ): Tensor[D] = {

    val so = if s.isDefined then s.get else new LongArrayRefOptional()
    val normStr =
      if norm.isDefined then new StringViewOptional(norm.get) else new StringViewOptional()
    val nt =
      if s.isDefined then torchNative.rfftn(input.native, s.get, dim, normStr)
      else
        torchNative.rfftn(
          input.native,
          new LongArrayRefOptional(),
          new LongArrayRefOptional(dim*),
          normStr
        )
    fromNative(nt)
  }

  def irfftn[D <: DType](
      input: Tensor[D],
      s: Option[Array[Long]] = None,
      dim: Array[Long] = Array(-2L, -1L),
      norm: Option[String] = None
  ): Tensor[D] = {

    val so = if s.isDefined then s.get else new LongArrayRefOptional()
//    val dimOp = new LongArrayRefOptional(dim*)
    val normStr =
      if norm.isDefined then new StringViewOptional(norm.get) else new StringViewOptional()
    val nt =
      if s.isDefined then torchNative.irfftn(input.native, s.get, dim, normStr)
      else
        torchNative.irfftn(
          input.native,
          new LongArrayRefOptional(),
          new LongArrayRefOptional(dim*),
          normStr
        )
    fromNative(nt)
  }

  def hfft2[D <: DType](
      input: Tensor[D],
      s: Option[Array[Long]] = None,
      dim: Array[Long] = Array(-2L, -1L),
      norm: Option[String] = None
  ): Tensor[D] = {

    val so = if s.isDefined then s.get else new LongArrayRefOptional()
    val normStr =
      if norm.isDefined then new StringViewOptional(norm.get) else new StringViewOptional()
    val nt =
      if s.isDefined then torchNative.hfft2(input.native, s.get, dim, normStr)
      else torchNative.hfft2(input.native, new LongArrayRefOptional(), dim, normStr)
    fromNative(nt)
  }

  def ihfft2[D <: DType](
      input: Tensor[D],
      s: Option[Array[Long]] = None,
      dim: Array[Long] = Array(-2L, -1L),
      norm: Option[String] = None
  ): Tensor[D] = {

    val so = if s.isDefined then s.get else new LongArrayRefOptional()
    val normStr =
      if norm.isDefined then new StringViewOptional(norm.get) else new StringViewOptional()
    val nt =
      if s.isDefined then torchNative.ihfft2(input.native, s.get, dim, normStr)
      else torchNative.ihfft2(input.native, new LongArrayRefOptional(), dim, normStr)
    fromNative(nt)
  }

  def hfftn[D <: DType](
      input: Tensor[D],
      s: Option[Array[Long]] = None,
      dim: Array[Long] = Array(-2L, -1L),
      norm: Option[String] = None
  ): Tensor[D] = {

    val so = if s.isDefined then s.get else new LongArrayRefOptional()
    val normStr =
      if norm.isDefined then new StringViewOptional(norm.get) else new StringViewOptional()
    val nt =
      if s.isDefined then torchNative.hfftn(input.native, s.get, dim, normStr)
      else torchNative.hfftn(input.native, new LongArrayRefOptional(), dim, normStr)
    fromNative(nt)
  }

  def ihfftn[D <: DType](
      input: Tensor[D],
      s: Option[Array[Long]] = None,
      dim: Array[Long] = Array(-2L, -1L),
      norm: Option[String] = None
  ): Tensor[D] = {

    val so = if s.isDefined then s.get else new LongArrayRefOptional()
    val normStr =
      if norm.isDefined then new StringViewOptional(norm.get) else new StringViewOptional()
    val nt =
      if s.isDefined then torchNative.ihfftn(input.native, s.get, dim, normStr)
      else torchNative.ihfftn(input.native, new LongArrayRefOptional(), dim, normStr)
    fromNative(nt)
  }

//  def irfft2[D <: DType](input: Tensor[D], s=None, dim=(-2, -1), norm=None, *, out=None): Tensor[D]

//  def rfftn[D <: DType](input: Tensor[D], s=None, dim=None, norm=None, *, out=None): Tensor[D]

//  def irfftn[D <: DType](input: Tensor[D], s=None, dim=None, norm=None, *, out=None): Tensor[D]

  def hfft[D <: DType](
      input: Tensor[D],
      n: Option[SymInt] = None,
      dim: Long = -1L,
      norm: Option[String] = None
  ): Tensor[D] = {
    val center = if n.isDefined then new SymIntOptional(n.get) else new SymIntOptional()
    val normStr =
      if norm.isDefined then new StringViewOptional(norm.get) else new StringViewOptional()
    fromNative(torchNative.hfft(input.native, center, dim, normStr))
  }

  def ihfft[D <: DType](
      input: Tensor[D],
      n: Option[SymInt] = None,
      dim: Long = -1L,
      norm: Option[String] = None
  ): Tensor[D] = {
    val center = if n.isDefined then new SymIntOptional(n.get) else new SymIntOptional()
    val normStr =
      if norm.isDefined then new StringViewOptional(norm.get) else new StringViewOptional()
    fromNative(torchNative.ihfft(input.native, center, dim, normStr))
  }

//  def hfft2[D <: DType](input: Tensor[D], s=None, dim=(-2, -1), norm=None, *, out=None): Tensor[D]

//  def ihfft2[D <: DType](input: Tensor[D], s=None, dim=(-2, -1), norm=None, *, out=None): Tensor[D]

//  def hfftn[D <: DType](input: Tensor[D], s=None, dim=None, norm=None, *, out=None): Tensor[D]

//  def ihfftn[D <: DType](input: Tensor[D], s=None, dim=None, norm=None, *, out=None): Tensor[D]

  object fft {
    def fftshift[D <: DType](input: Tensor[D], dim: Seq[Long]): Tensor[D] = fromNative(
      torchNative.fftshift(input.native, dim*)
    )

    def ifftshift[D <: DType](input: Tensor[D], dim: Seq[Long]): Tensor[D] = fromNative(
      torchNative.ifftshift(input.native, dim*)
    )

    def fft[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.fft(input.native))

    def ifft[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.ifft(input.native))

    def fft2[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.fft2(input.native))

    def ifft2[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.ifft2(input.native))

    def fftn[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.fftn(input.native))

    def ifftn[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.ifftn(input.native))

    def rfft[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.rfft(input.native))

    def irfft[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.irfft(input.native))

    def rfft2[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.rfft2(input.native))

    def irfft2[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(
      torchNative.irfft2(input.native)
    )

    def rfftn[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.rfftn(input.native))

    def irfftn[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(
      torchNative.irfftn(input.native)
    )

    def hfft[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.hfft(input.native))

    def ihfft[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.ihfft(input.native))

    def hfft2[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.hfft2(input.native))

    def ihfft2[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(
      torchNative.ihfft2(input.native)
    )

    def hfftn[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.hfftn(input.native))

    def ihfftn[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(
      torchNative.ihfftn(input.native)
    )

    def fftshift[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(
      torchNative.fftshift(input.native)
    )

    def ifftshift[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(
      torchNative.ifftshift(input.native)
    )

    def fftfreq[D <: DType](n: Long): Tensor[D] = fromNative(torchNative.fftfreq(n))

    def fftfreq[D <: DType](n: Long, d: Double): Tensor[D] = fromNative(torchNative.fftfreq(n, d))

  }

  def fftfreq[D <: DType](n: Long, out: TensorOptions): Tensor[D] = fromNative(
    torchNative.fftfreq(n, out)
  )

  def fftfreq[D <: DType](n: Long, d: Double, out: TensorOptions): Tensor[D] = fromNative(
    torchNative.fftfreq(n, d, out)
  )

  //  def fftfreq[D <: DType](n: Long, d: Double, out:): Tensor[D] = fromNative(torchNative.fftfreq(n, d ,out))

//  def fftfreq[D <: DType](input: Tensor[D], n, d = 1.0, out = None, dtype = None, layout = torch.strided, device = None, requires_grad = False): Tensor[D]

//  def rfftfreq[D <: DType](input: Tensor[D], n, d = 1.0,  out = None, dtype = None, layout = torch.strided, device = None, requires_grad = False): Tensor[D]

  def rfftfreq[D <: DType](n: Long, out: TensorOptions): Tensor[D] = fromNative(
    torchNative.rfftfreq(n, out)
  )

  def rfftfreq[D <: DType](n: Long, d: Double, out: TensorOptions): Tensor[D] = fromNative(
    torchNative.rfftfreq(n, d, out)
  )

  def torch_fft_fftfreq[D <: DType](n: Long, d: Double, out: TensorOptions): Tensor[D] = fromNative(
    torchNative.torch_fft_fftfreq(n, d, out)
  )

  def torch_fft_fftfreq[D <: DType](n: Long): Tensor[D] = fromNative(
    torchNative.torch_fft_fftfreq(n)
  )

  def torch_fft_rfftfreq[D <: DType](n: Long, d: Double, out: TensorOptions): Tensor[D] =
    fromNative(torchNative.torch_fft_rfftfreq(n, d, out))

  def torch_fft_rfftfreq[D <: DType](n: Long): Tensor[D] = fromNative(
    torchNative.torch_fft_rfftfreq(n)
  )

  //  def fft[D <: DType](input: Tensor[D]): Tensor[D]
//
//  def fft[D <: DType](input: Tensor[D]): Tensor[D]
//
}
