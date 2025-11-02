package torch
package ops

import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.fromNative
import org.bytedeco.pytorch.{BoolOptional, DeviceOptional, LayoutOptional, ScalarTypeOptional}
import torch.Device.CPU
import torch.Derive
import torch.Layout.{Strided, Sparse, SparseCsc, SparseBsc, SparseBsr, SparseCsr}
trait SparseOps {

  // https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html
  def sparse_coo_tensor[D1 <: DType, D2 <: SparseIntNN](
      indices: Tensor[D2],
      values: Tensor[D1]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.sparse_coo_tensor(indices.native, values.native))

  def sparse_coo_tensor[D1 <: DType, D2 <: SparseIntNN](
      indices: Tensor[D2],
      values: Tensor[D1],
      sizeSeq: Seq[Int]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.sparse_coo_tensor(indices.native, values.native, sizeSeq.map(_.toLong)*))

  def sparse_coo_tensor[D1 <: DType, D2 <: SparseIntNN](
      indices: Tensor[D2],
      values: Tensor[D1],
      dtype: DType | Option[DType] = DType.float32,
      layout: Layout = Sparse,
      device: Device = CPU,
      requires_grad: Boolean = false
  ): Tensor[Promoted[D1, D2]] = {
    val derivedDType = dtype match
      case None             => values.dtype
      case d: Option[DType] => d.get
      case d: DType         => d
    val scalarNative =
      if dtype == values.dtype then ScalarTypeOptional()
      else ScalarTypeOptional(derivedDType.toScalarType)
    val layoutNative: LayoutOptional = LayoutOptional(layout.toNative)
    val deviceNative: DeviceOptional = DeviceOptional(device.toNative)
    val boolOption = BoolOptional(requires_grad)
    fromNative(
      torchNative.sparse_coo_tensor(
        indices.native,
        values.native,
        scalarNative,
        layoutNative,
        deviceNative,
        boolOption,
        boolOption
      )
    )
  }

  // https://pytorch.org/docs/stable/generated/torch.sparse_csc_tensor.html
  def sparse_csc_tensor[D1 <: DType, D2 <: SparseIntNN](
      ccol_indices: Tensor[D2],
      row_indices: Tensor[D2],
      values: Tensor[D1],
      size: Seq[Int],
      dtype: DType | Option[DType] = DType.float32,
      layout: Layout = SparseCsc,
      device: Device = CPU,
      requires_grad: Boolean = false
  ): Tensor[Promoted[D1, D2]] = {
    val derivedDType = dtype match
      case None             => values.dtype
      case d: Option[DType] => d.get
      case d: DType         => d
    val scalarNative =
      if dtype == values.dtype then ScalarTypeOptional()
      else ScalarTypeOptional(derivedDType.toScalarType)
    val layoutNative: LayoutOptional = LayoutOptional(layout.toNative)
    val deviceNative: DeviceOptional = DeviceOptional(device.toNative)
    val boolOption = BoolOptional(requires_grad)
    fromNative(
      torchNative.sparse_csc_tensor(
        ccol_indices.native,
        row_indices.native,
        values.native,
        size.map(_.toLong).toArray,
        scalarNative,
        layoutNative,
        deviceNative,
        boolOption
      )
    )
  }

  def is_sparse[D1 <: DType](tensor: Tensor[D1]): Boolean = {
    val sparseLayout = Array(Sparse, SparseCsc, SparseBsc, SparseBsr, SparseCsr)
    if sparseLayout.contains(tensor.layout) then true else false
  }

  def to_dense[D1 <: DType](tensor: Tensor[D1]): Tensor[D1] = {
    fromNative(tensor.native.to_dense)
  }

  // https://pytorch.org/docs/stable/generated/torch.sparse_csr_tensor.html
  def sparse_csr_tensor[D1 <: DType, D2 <: SparseIntNN](
      crow_indices: Tensor[D2],
      col_indices: Tensor[D2],
      values: Tensor[D1],
      dtype: DType | Option[DType] = DType.float32,
      layout: Layout = SparseCsr,
      device: Device = CPU,
      requires_grad: Boolean = false
  ): Tensor[Promoted[D1, D2]] = {
    val derivedDType = dtype match
      case None             => values.dtype
      case d: Option[DType] => d.get
      case d: DType         => d
    val scalarNative =
      if dtype == values.dtype then ScalarTypeOptional()
      else ScalarTypeOptional(derivedDType.toScalarType)
    val layoutNative: LayoutOptional = LayoutOptional(layout.toNative)
    val deviceNative: DeviceOptional = DeviceOptional(device.toNative)
    val boolOption = BoolOptional(requires_grad)

    fromNative(
      torchNative.sparse_csr_tensor(
        crow_indices.native,
        col_indices.native,
        values.native,
        scalarNative,
        layoutNative,
        deviceNative,
        boolOption
      )
    )
  }

  // https://pytorch.org/docs/stable/generated/torch.sparse_bsc_tensor.html
  def sparse_bsc_tensor[D1 <: DType, D2 <: SparseIntNN](
      ccol_indices: Tensor[D2],
      row_indices: Tensor[D2],
      values: Tensor[D1],
      size: Seq[Int],
      dtype: DType | Option[DType] = DType.float32,
      layout: Layout = SparseBsc,
      device: Device = CPU,
      requires_grad: Boolean = false
  ): Tensor[Promoted[D1, D2]] = {

    val derivedDType = dtype match
      case None             => values.dtype
      case d: Option[DType] => d.get
      case d: DType         => d
    val scalarNative =
      if dtype == values.dtype then ScalarTypeOptional()
      else ScalarTypeOptional(derivedDType.toScalarType)
    val layoutNative: LayoutOptional = LayoutOptional(layout.toNative)
    val deviceNative: DeviceOptional = DeviceOptional(device.toNative)
    val boolOption = BoolOptional(requires_grad)
    fromNative(
      torchNative.sparse_bsc_tensor(
        ccol_indices.native,
        row_indices.native,
        values.native,
        size.map(_.toLong).toArray,
        scalarNative,
        layoutNative,
        deviceNative,
        boolOption
      )
    )

  }

  //    https: //pytorch.org/docs/stable/generated/torch.sparse_bsr_tensor.html
  def sparse_bsr_tensor[D1 <: DType, D2 <: SparseIntNN](
      crow_indices: Tensor[D2],
      col_indices: Tensor[D2],
      values: Tensor[D1],
      size: Seq[Int],
      dtype: DType | Option[DType] = DType.float32,
      layout: Layout = SparseBsr,
      device: Device = CPU,
      requires_grad: Boolean = false
  ): Tensor[Promoted[D1, D2]] = {
    val layoutNative: LayoutOptional = LayoutOptional(layout.toNative)
    val deviceNative: DeviceOptional = DeviceOptional(device.toNative)
    val boolOption = BoolOptional(requires_grad)
    val derivedDType = dtype match
      case None             => values.dtype
      case d: Option[DType] => d.get
      case d: DType         => d
    val scalarNative =
      if dtype == values.dtype then ScalarTypeOptional()
      else ScalarTypeOptional(derivedDType.toScalarType)
    fromNative(
      torchNative.sparse_bsr_tensor(
        crow_indices.native,
        col_indices.native,
        values.native,
        size.map(_.toLong).toArray,
        scalarNative,
        layoutNative,
        deviceNative,
        boolOption
      )
    )
  }

  // https://pytorch.org/docs/stable/generated/torch.sparse_compressed_tensor.html
  def sparse_compressed_tensor[D1 <: DType, D2 <: SparseIntNN](
      compressed_indices: Tensor[D2],
      plain_indices: Tensor[D2],
      values: Tensor[D1],
      size: Seq[Int],
      dtype: DType | Option[DType] = DType.float32,
      layout: Layout = Sparse,
      device: Device = CPU,
      requires_grad: Boolean = false
  ): Tensor[Promoted[D1, D2]] = {
    val derivedDType = dtype match
      case None             => values.dtype
      case d: Option[DType] => d.get
      case d: DType         => d
    val scalarNative =
      if dtype == values.dtype then ScalarTypeOptional()
      else ScalarTypeOptional(derivedDType.toScalarType)
    val layoutNative: LayoutOptional = LayoutOptional(layout.toNative)
    val deviceNative: DeviceOptional = DeviceOptional(device.toNative)
    val boolOption = BoolOptional(requires_grad)
    fromNative(
      torchNative.sparse_compressed_tensor(
        compressed_indices.native,
        plain_indices.native,
        values.native,
        size.map(_.toLong).toArray,
        scalarNative,
        layoutNative,
        deviceNative,
        boolOption
      )
    )
  }

  def sparse_mask_out[D1 <: DType, D2 <: DType](
      t1: Tensor[D1],
      t2: Tensor[D2],
      t3: Tensor[D1 | D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.sparse_mask_out(t1.native, t2.native, t3.native))

}
