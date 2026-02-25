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

import torch.Layout.Strided
import torch.Device.CPU
import torch.internal.NativeConverters
import NativeConverters.*
import org.bytedeco.pytorch.TensorOptions
import org.bytedeco.pytorch.global.torch as torchNative
import torch.{Float32, Int64}

/** Random Sampling
  *
  * https://pytorch.org/docs/stable/torch.html#random-sampling
  */
private[torch] trait RandomSamplingOps {

  val normalfDtype: Float32 = float32
  val normaliDtype: Int64 = int64
  /* Returns a tensor where each row contains `numSamples` indices sampled from the multinomial probability distribution located in the corresponding row of tensor `input`. */
  def multinomial[D <: FloatNN](
      input: Tensor[D],
      numSamples: Long,
      replacement: Boolean = false,
      generator: Option[Generator] | Generator = None
  ): Tensor[Int64] =
    fromNative(torchNative.multinomial(input.native, numSamples, replacement, generator.toOptional))

  def uniform[D1 <: DType](t1: Tensor[D1]): Tensor[D1] = {
    fromNative(torchNative.uniform(t1.native))
  }

  def uniform[D1 <: DType](
      low: Tensor[D1],
      mean: Double,
      std: Double,
      generator: Option[Generator] | Generator = None
  ): Tensor[D1] = {
    fromNative(torchNative.uniform(low.native, mean, std, generator.toOptional))
  }

  def poisson[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.poisson(t1.native))

  def poisson[D1 <: DType, D2 <: DType](
      lambda: Tensor[D1],
      generator: Option[Generator] | Generator = None
  ): Tensor[D1] =
    fromNative(torchNative.poisson(lambda.native, generator.toOptional))

  def normal[D1 <: DType, D2 <: DType](
      mean: Tensor[D1],
      std: Tensor[D2],
      generator: Option[Generator] | Generator
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.normal(mean.native, std.native, generator.toOptional))

  def normal[D1 <: DType](mean: Double, std: Double, size: Array[Long]): Tensor[D1] =
    fromNative(torchNative.normal(mean, std, size*))

  def torch_normal[D1 <: DType](mean: Double, std: Double, size: Array[Long]): Tensor[D1] =
    fromNative(torchNative.torch_normal(mean, std, size*))

  def torch_normal[D1 <: DType](
      mean: Double,
      std: Double,
      size: Array[Long],
      generator: Option[Generator] | Generator,
      dtype: D1 = int64,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false
  ): Tensor[D1] = {
    val options = NativeConverters.tensorOptions(dtype, layout, device, requires_grad)
    fromNative(torchNative.torch_normal(mean, std, size, generator.toOptional, options))
  }

  def log_normal[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.log_normal(t1.native))

  def log_normal[D1 <: DType](
      mean: Double,
      std: Double,
      size: Tensor[D1],
      generator: Option[Generator] | Generator
  ): Tensor[D1] =
    fromNative(torchNative.log_normal(size.native, mean, std, generator.toOptional))

  def normal[D1 <: DType](
      mean: Double,
      std: Double,
      size: Array[Long],
      generator: Option[Generator] | Generator,
      options: TensorOptions
  ): Tensor[D1] =
    fromNative(torchNative.normal(mean, std, size, generator.toOptional, options))

  def normal[D1 <: DType](
      mean: Double,
      std: Double,
      size: Array[Long],
      generator: Option[Generator] | Generator,
      dtype: D1 = int64,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false
  ): Tensor[D1] = {
    val options = NativeConverters.tensorOptions(dtype, layout, device, requires_grad)
    fromNative(torchNative.normal(mean, std, size, generator.toOptional, options))
  }

  // TODO normal Returns a tensor of random numbers drawn from separate normal distributions whose mean and standard deviation are given.
// TODO poisson Returns a tensor of the same size as input with each element sampled from a Poisson distribution with rate parameter given by the corresponding element in input i.e.,

  /** Returns a tensor filled with random numbers from a uniform distribution on the interval
    * `[0,1)`
    *
    * The shape of the tensor is defined by the variable argument `size`.
    *
    * @param size
    *   a sequence of integers defining the shape of the output tensor.
    * @param dtype
    *   the desired data type of returned tensor.
    * @param layout
    *   the desired layout of returned Tensor.
    * @param device
    *   the desired device of returned tensor.
    * @param requiresGrad
    *   If autograd should record operations on the returned tensor.
    * @tparam T
    *   the dtype of the created tensor.
    */
  def rand_raw[D <: FloatNN | ComplexNN](
      size: Int*
  )(using requires_grad: Boolean = false)(using dtypes: D = float32): Tensor[D] = {
    rand(size.toSeq, dtypes, requires_grad, Strided, CPU)
  }

  def rands[D <: FloatNN | ComplexNN](size: Int*)(implicit dtype: D = float32): Tensor[D] = {
    rand(size = size.toSeq, dtype = dtype, requires_grad = false, layout = Strided, device = CPU)
  }

  def rand[D <: FloatNN | ComplexNN](
      s1: Int,
      requires_grad: Boolean,
      dtype: D
  ): Tensor[D] = {
    fromNative(
      torchNative.torch_rand(
        Seq(s1).toArray.map(_.toLong),
        NativeConverters.tensorOptions(dtype, Strided, CPU, requires_grad)
      )
    )
  }
  def rand[D <: FloatNN | ComplexNN](
      s1: Int,
      s2: Int,
      requires_grad: Boolean,
      dtype: D
  ): Tensor[D] = {
    fromNative(
      torchNative.torch_rand(
        Seq(s1, s2).toArray.map(_.toLong),
        NativeConverters.tensorOptions(dtype, Strided, CPU, requires_grad)
      )
    )
  }
  def rand[D <: FloatNN | ComplexNN](
      s1: Int,
      s2: Int,
      s3: Int,
      requires_grad: Boolean,
      dtype: D
  ): Tensor[D] = {
    fromNative(
      torchNative.torch_rand(
        Seq(s1, s2, s3).toArray.map(_.toLong),
        NativeConverters.tensorOptions(dtype, Strided, CPU, requires_grad)
      )
    )
  }
  def rand[D <: FloatNN | ComplexNN](
      s1: Int,
      s2: Int,
      s3: Int,
      s4: Int,
      requires_grad: Boolean,
      dtype: D
  ): Tensor[D] = {

    fromNative(
      torchNative.torch_rand(
        Seq(s1, s2, s3, s4).toArray.map(_.toLong),
        NativeConverters.tensorOptions(dtype, Strided, CPU, requires_grad)
      )
    )
  }

  def rand[D <: FloatNN | ComplexNN](
      s1: Int,
      s2: Int,
      s3: Int,
      s4: Int,
      s5: Int,
      requires_grad: Boolean,
      dtype: D
  ): Tensor[D] = {

    fromNative(
      torchNative.torch_rand(
        Seq(s1, s2, s3, s4, s5).toArray.map(_.toLong),
        NativeConverters.tensorOptions(dtype, Strided, CPU, requires_grad)
      )
    )
  }
  def rand[D <: FloatNN | ComplexNN](
      size: Seq[Int],
      dtype: D = float32,
      requires_grad: Boolean = false,
      layout: Layout = Strided,
      device: Device = CPU
  ): Tensor[D] =
    fromNative(
      torchNative.torch_rand(
        size.toArray.map(_.toLong),
        NativeConverters.tensorOptions(dtype, layout, device, requires_grad)
      )
    )

  def rand[D <: FloatNN | ComplexNN](
      size: Int*
  ): Tensor[D] =
    fromNative(
      torchNative.torch_rand(
        size.toArray.map(_.toLong),
        NativeConverters.tensorOptions(float32, Strided, CPU, false)
      )
    )

  /** Returns a tensor with the same size as `input` that is filled with random numbers from a
    * uniform distribution on the interval $[0, 1)$.
    *
    * `torch.randLike(input)` is equivalent to `torch.rand(input.size(), dtype=input.dtype,
    * layout=input.layout, device=input.device)`.
    *
    * @param input
    *   the size of `input` will determine size of the output tensor.
    * @param dtype
    *   the desired data type of returned Tensor. If `derive`, defaults to the dtype of `input`.
    * @param layout
    *   the desired layout of returned tensor. If `derive`, defaults to the layout of `input`.
    * @param device
    *   the desired device of returned tensor. If `derive` , defaults to the device of `input`.
    * @param requiresGrad
    *   If autograd should record operations on the returned tensor.
    * @param memoryFormat
    *   the desired memory format of returned Tensor.
    */
  def randLike[D <: DType, D2 <: DType | Derive](
      input: Tensor[D],
      dtype: D2 = derive,
      layout: Layout | Derive = derive,
      device: Device | Derive = derive,
      requires_grad: Boolean = false,
      memoryFormat: MemoryFormat = MemoryFormat.Preserve
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    xLike(input, dtype, layout, device, requires_grad, memoryFormat, torchNative.torch_rand_like)

  def rand_like[D <: DType, D2 <: DType | Derive](
      input: Tensor[D],
      dtype: D2 = derive,
      layout: Layout | Derive = derive,
      device: Device | Derive = derive,
      requires_grad: Boolean = false,
      memoryFormat: MemoryFormat = MemoryFormat.Preserve
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    xLike(input, dtype, layout, device, requires_grad, memoryFormat, torchNative.torch_rand_like)

  /** Returns a tensor filled with random integers generated uniformly between `low` (inclusive) and
    * `high` (exclusive).
    *
    * The shape of the tensor is defined by the variable argument `size`.
    *
    * @param low
    *   Lowest integer to be drawn from the distribution. Default: 0.
    * @param high
    *   One above the highest integer to be drawn from the distribution.
    * @param size
    *   a tuple defining the shape of the output tensor.
    * @param generator
    *   a pseudorandom number generator for sampling
    * @param dtype
    *   the desired data type of returned tensor.
    * @param layout
    *   the desired layout of returned Tensor.
    * @param device
    *   the desired device of returned tensor.
    * @param requiresGrad
    *   If autograd should record operations on the returned tensor.
    * @tparam T
    *   the dtype of the created tensor.
    */
  def randint_none[D <: DType](low: Long, high: Long, size: Int*)(using
      requires_grad: Boolean = false
  )(using dtype: D = int64): Tensor[D] = {
    randint(
      low = low,
      high = high,
      size = size.toSeq,
      generator = None,
      dtype = dtype,
      layout = Strided,
      device = CPU,
      requires_grad = requires_grad
    )
  }

  def randints[D <: DType](low: Long, high: Long, sizes: Int*)(implicit
      dtype: D = float32,
      requires_grad: Boolean = false
  ): Tensor[D] = {
    randint(
      low = low,
      high = high,
      size = sizes.toSeq,
      generator = None,
      dtype = dtype,
      layout = Strided,
      device = CPU,
      requires_grad = requires_grad
    )
  }
  def randint[D <: DType](
      low: Long = 0,
      high: Long,
      size: Seq[Int],
      generator: Option[Generator] | Generator = None,
      dtype: D = int64,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false
  ): Tensor[D] =
    fromNative(
      torchNative.torch_randint(
        low,
        high,
        size.toArray.map(_.toLong),
        generator.toOptional,
        NativeConverters.tensorOptions(dtype, layout, device, requires_grad)
      )
    )

  def randint_raws[D <: DType](low: Long, high: Long, size: Int*): Tensor[D] =
    val generator = new Generator(CPU)
    fromNative(
      torchNative.torch_randint(
        low,
        high,
        size.toArray.map(_.toLong),
        generator.toNativeOptional,
        NativeConverters.tensorOptions(int64, Strided, CPU, false)
      )
    )
// TODO randint_like Returns a tensor with the same shape as Tensor input filled with random integers generated uniformly between low (inclusive) and high (exclusive).

// TODO Randnd acepts Seq[Int] | Int

  def randn_raw[D <: FloatNN | ComplexNN](
      size: Int*
  )(using requires_grad: Boolean = false)(using dtype: D = float32): Tensor[D] = {
    randn(
      size = size.toSeq,
      dtype = dtype,
      layout = Strided,
      device = CPU,
      requires_grad = requires_grad
    )
  }

  def randns[D <: FloatNN | ComplexNN](size: Int*)(implicit dtype: D = float32): Tensor[D] = {
    randn(
      size = size.toSeq,
      dtype = dtype,
      layout = Strided,
      device = CPU,
      requires_grad = false
    )
  }

  def randn[D <: FloatNN | ComplexNN](
      s1: Int,
      requires_grad: Boolean,
      dtype: D
  ): Tensor[D] = {
    fromNative(
      torchNative.torch_randn(
        Seq(s1).toArray.map(_.toLong),
        NativeConverters.tensorOptions(dtype, Strided, CPU, requires_grad)
      )
    )
  }

  def randn[D <: FloatNN | ComplexNN](
      s1: Int,
      s2: Int,
      requires_grad: Boolean,
      dtype: D
  ): Tensor[D] = {
    fromNative(
      torchNative.torch_randn(
        Seq(s1, s2).toArray.map(_.toLong),
        NativeConverters.tensorOptions(dtype, Strided, CPU, requires_grad)
      )
    )
  }

  def randn[D <: FloatNN | ComplexNN](
      s1: Int,
      s2: Int,
      s3: Int,
      requires_grad: Boolean
  ): Tensor[D] = {
    fromNative(
      torchNative.torch_randn(
        Seq(s1, s2, s3).toArray.map(_.toLong),
        NativeConverters.tensorOptions(float32, Strided, CPU, requires_grad)
      )
    )
  }

  def randn[D <: FloatNN | ComplexNN](
      s1: Int,
      s2: Int,
      s3: Int,
      s4: Int,
      requires_grad: Boolean
  ): Tensor[D] = {

    fromNative(
      torchNative.torch_randn(
        Seq(s1, s2, s3, s4).toArray.map(_.toLong),
        NativeConverters.tensorOptions(float32, Strided, CPU, requires_grad)
      )
    )
  }

  def randn[D <: FloatNN | ComplexNN](
      s1: Int,
      s2: Int,
      s3: Int,
      s4: Int,
      s5: Int,
      requires_grad: Boolean,
      dtype: D
  ): Tensor[D] = {

    fromNative(
      torchNative.torch_randn(
        Seq(s1, s2, s3, s4, s5).toArray.map(_.toLong),
        NativeConverters.tensorOptions(dtype, Strided, CPU, requires_grad)
      )
    )
  }

  def randn[D <: FloatNN | ComplexNN](
      size: Seq[Int],
      dtype: D = float32,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false
  ): Tensor[D] =
    fromNative(
      torchNative.torch_randn(
        size.toArray.map(_.toLong),
        NativeConverters.tensorOptions(dtype, layout, device, requires_grad)
      )
    )

  def randn[D <: FloatNN | ComplexNN](
      size: Int*
  ): Tensor[D] =
    fromNative(
      torchNative.torch_randn(
        size.toArray.map(_.toLong),
        NativeConverters.tensorOptions(float32, Strided, CPU, false)
      )
    )
// TODO randn_like Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1.

  /** Returns a random permutation of integers from 0 to n - 1.
    *
    * TODO support custom generator
    */
  def randperm_torch[D <: DType](
      n: Long,
      dtype: D = int64,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false,
      pinMemory: Boolean = false
  ): Tensor[D] =
    fromNative(
      torchNative.torch_randperm(
        n,
        NativeConverters.tensorOptions(dtype, layout, device, requires_grad, pinMemory)
      )
    )

  def randperm[D <: DType](
      n: Long,
      generator: Option[Generator] | Generator = None,
      dtype: D = int64,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false,
      pinMemory: Boolean = false
  ): Tensor[D] =
    fromNative(
      torchNative.torch_randperm(
        n,
        generator.toOptional,
        NativeConverters.tensorOptions(dtype, layout, device, requires_grad, pinMemory)
      )
    )

  def bernoulli[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.bernoulli(t1.native))

  def bernoulli[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2] | Long,
      generator: Option[Generator] | Generator = None
  ): Tensor[Promoted[D1, D2]] = {
    t2 match {
      case t: Tensor[D2] =>
        if generator != None then
          fromNative(torchNative.bernoulli(t1.native, t.native, generator.toOptional))
        else fromNative(torchNative.bernoulli(t1.native, t.native))
      case l: Long =>
        if generator != None then
          fromNative(torchNative.bernoulli(t1.native, l, generator.toOptional))
        else fromNative(torchNative.bernoulli(t1.native, l))

    }

  }

  def bernoulli[D1 <: DType](
      t1: Tensor[D1],
      generator: Option[Generator]
  ): Tensor[D1] = {
    if generator.isDefined then
      fromNative(torchNative.bernoulli(t1.native, generator.get.toOptional))
    else fromNative(torchNative.bernoulli(t1.native))

  }

  def binomial[D1 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D1],
      generator: Option[Generator] | Generator = None
  ): Tensor[D1] = {
    if generator != None then
      fromNative(torchNative.binomial(t1.native, t2.native, generator.toOptional))
    else fromNative(torchNative.binomial(t1.native, t2.native))

  }

  def cauchy[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.cauchy(t1.native))

  def cauchy[D1 <: DType](
      t1: Tensor[D1],
      median: Double = 0,
      sigma: Double = 1,
      generator: Option[Generator] | Generator = None
  ): Tensor[D1] = {
    fromNative(torchNative.cauchy(t1.native, median, sigma, generator.toOptional))

  }

  def exponential[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.exponential(t1.native))

  def exponential[D1 <: DType](
      t1: Tensor[D1],
      lambd: Double = 1,
      generator: Option[Generator] | Generator = None
  ): Tensor[D1] = {
    fromNative(torchNative.exponential(t1.native, lambd, generator.toOptional))

  }

  def geometric[D1 <: DType](
      t1: Tensor[D1],
      p: Double = 0.5,
      generator: Option[Generator] | Generator = None
  ): Tensor[D1] = {
    fromNative(torchNative.geometric(t1.native, p, generator.toOptional))

  }

  def rrelu[D1 <: DType, S <: ScalaType](
      t1: Tensor[D1],
      lower: S = 0.125,
      upper: S = 0.3333333333333333,
      train: Boolean = false,
      generator: Option[Generator] | Generator = None
  ): Tensor[D1] =
    fromNative(
      torchNative.rrelu(t1.native, toScalar(lower), toScalar(upper), train, generator.toOptional)
    )

  def rrelu_with_noise[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.rrelu_with_noise(t1.native, t2.native))

  def rrelu_with_noise[D1 <: DType, D2 <: DType, S <: ScalaType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      lower: S = 0.125,
      upper: S = 0.3333333333333333,
      train: Boolean = false,
      generator: Option[Generator] | Generator = None
  ): Tensor[Promoted[D1, D2]] =
    fromNative(
      torchNative.rrelu_with_noise(
        t1.native,
        t2.native,
        toScalar(lower),
        toScalar(upper),
        train,
        generator.toOptional
      )
    )

  def rrelu_[D1 <: DType, S <: ScalaType](
      t1: Tensor[D1],
      lower: S = 0.125,
      upper: S = 0.3333333333333333,
      train: Boolean = false,
      generator: Option[Generator] | Generator = None
  ): Tensor[D1] =
    fromNative(
      torchNative.rrelu_(t1.native, toScalar(lower), toScalar(upper), train, generator.toOptional)
    )

  def rrelu_with_noise_[D1 <: DType, D2 <: DType, S <: ScalaType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      lower: S = 0.125,
      upper: S = 0.3333333333333333,
      train: Boolean = false,
      generator: Option[Generator] | Generator = None
  ): Tensor[Promoted[D1, D2]] =
    fromNative(
      torchNative.rrelu_with_noise_(
        t1.native,
        t2.native,
        toScalar(lower),
        toScalar(upper),
        train,
        generator.toOptional
      )
    )

  // def clip[D1 <: DType, S <: ScalaType](t1: Tensor[D1], min: S): Tensor[Div[D1, ScalaToDType[S]]] =
  //    fromNative(torchNative.clip(t1.native, new ScalarOptional(toScalar(min))))
  //  public static native Tensor rrelu(@Const @ByRef Tensor var0,
  //  @Const @ByRef(nullValue = "at::Scalar(0.125)") Scalar var1,
  //  @Const @ByRef(nullValue = "at::Scalar(0.3333333333333333)")
  //  Scalar var2, @Cast({"bool"}) boolean var3, @ByVal(nullValue = "std::optional<at::Generator>(
  //  ::std::nullopt)") GeneratorOptional var4);

  // public static native Tensor rrelu_with_noise(@Const @ByRef Tensor var0, @ByRef Tensor var1,
  // @Const @ByRef(nullValue = "at::Scalar(0.125)") Scalar var2,
  // @Const @ByRef(nullValue = "at::Scalar(0.3333333333333333)") Scalar var3,
  // @Cast({"bool"}) boolean var4,
  // @ByVal(nullValue = "std::optional<at::Generator>(::std::nullopt)") GeneratorOptional var5);

  def manualSeed(seed: Long) = torchNative.manual_seed(seed)

  def manual_seed(seed: Long) = torchNative.manual_seed(seed)

  def setNumThreads(threads: Int): Unit = torchNative.set_num_threads(threads)

  def set_num_threads(threads: Int): Unit = torchNative.set_num_threads(threads)

  def create_cpu_generator = torchNative.createCPUGenerator()

  def create_cpu_generator(seed: Long) = torchNative.createCPUGenerator(seed)

  def get_default_cpu_generator = torchNative.getDefaultCPUGenerator()

  def createCPUGenerator = torchNative.createCPUGenerator()

  def createCPUGenerator(seed: Long) = torchNative.createCPUGenerator(seed)

  def getDefaultCPUGenerator = torchNative.getDefaultCPUGenerator()

  //     public static native Generator getDefaultCPUGenerator();
  //
  //    @Namespace("at::detail")
  //    @ByVal
  //    public static native Generator createCPUGenerator(@Cast({"uint64_t"}) long var0);
  //
  //    @Namespace("at::detail")
  //    @ByVal
  //    public static native Generator createCPUGenerator();
}

// TODO seed Sets the seed for generating random numbers to a non-deterministic random number.
// TODO manual_seed Sets the seed for generating random numbers.
// TODO initial_seed Returns the initial seed for generating random numbers as a Python long.
// TODO get_rng_state Returns the random number generator state as a torch.ByteTensor.
// TODO set_rng_state Sets the random number generator state.
// TODO bernoulli Draws binary random numbers (0 or 1) from a Bernoulli distribution.
