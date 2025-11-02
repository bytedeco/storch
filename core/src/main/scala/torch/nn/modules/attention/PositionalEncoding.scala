package torch
package nn
package modules
package attention

//import torch.nn.modules.{HasParams, Module}
//import torch.{BFloat16, Default, Float32, Tensor}
//import torch.internal.NativeConverters.{fromNative, toNative}

class PositionalEncoding[ParamType <: FloatNN | ComplexNN: Default](
    val d_model: Long,
    val max_len: Long = 28 * 28
) extends HasParams[ParamType]
    with TensorModule[ParamType] {
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  val arr = Seq(max_len, d_model)
  var encoding = torch.zeros(size = arr.map(_.toInt), dtype = this.paramType)
  val position = torch.arange(0, max_len, dtype = this.paramType).unsqueeze(1)
  val div_term =
    torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(Tensor(10000.0)) / d_model))
  val sinPosition = torch.sin(position * div_term).to(dtype = this.paramType)
  val cosPosition = torch.cos(position * div_term).to(dtype = this.paramType)
  val indexSin = torch.Tensor(Seq(0L, 1L))
  val indexCos = torch.Tensor(Seq(1L, 1L))
  encoding.update(indices = Seq(::, 0.::(2)), values = sinPosition)
  encoding.update(indices = Seq(---, 1.::(2)), values = cosPosition)
  encoding = encoding.unsqueeze(0)

  override def toString =
    s"${getClass.getSimpleName}(d_model=$d_model max_len=$max_len)"

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {

    forward(input)
  }

  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    x.add(encoding.index(::, 0.&&(x.size(1))).to(x.device))
  }

}

object PositionalEncoding {
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      d_model: Long,
      max_len: Long = 28 * 28
  ): PositionalEncoding[ParamType] = {
    new PositionalEncoding[ParamType](d_model, max_len)
  }
}
