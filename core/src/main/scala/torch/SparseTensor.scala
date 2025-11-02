//package torch
//
//import ai.storch.Tensor
//import scala.language.implicitConversions
//
//// 定义稀疏布局的枚举
//enum SparseLayout:
//  case COO, CSR, CSC, BSR, BSC
//
//class SparseTensor[T](
//    val indices: Tensor[Long],
//    val values: Tensor[T],
//    val size: Seq[Int],
//    val nnz: Int,
//    val layout: SparseLayout
//):
//
//  // 转换为 CSR 格式
//  def to_sparse_csr: SparseTensor[T] =
//    val numRows = size(0)
//    val csrIndptr = Tensor.zeros[Long](numRows + 1)
//    val csrIndices = Tensor.zeros[Long](nnz)
//    val csrValues = Tensor.zeros[T](nnz)
//    var ptr = 0
//    for (i <- 0 until numRows)
//      csrIndptr(i) = ptr
//      for (j <- 0 until nnz)
//        if (indices(0, j) == i)
//          csrIndices(ptr) = indices(1, j)
//          csrValues(ptr) = values(j)
//          ptr += 1
//    csrIndptr(numRows) = ptr
//    new SparseTensor(Tensor.stack(Seq(csrIndptr, csrIndices)), csrValues, size, nnz, SparseLayout.CSR)
//
//  // 转换为 COO 格式
//  def to_sparse_coo: SparseTensor[T] =
//    if layout == SparseLayout.COO then this
//    else
//      layout match
//        case SparseLayout.CSR =>
//          val numRows = size(0)
//          val cooIndices = Tensor.zeros[Long](2, nnz)
//          val cooValues = Tensor.zeros[T](nnz)
//          val indptr = indices.select(0, *)
//          val csrIndices = indices.select(1, *)
//          var ptr = 0
//          for (i <- 0 until numRows)
//            val start = indptr(i).toInt
//            val end = indptr(i + 1).toInt
//            for (j <- start until end)
//              cooIndices(0, ptr) = i
//              cooIndices(1, ptr) = csrIndices(j)
//              cooValues(ptr) = values(j)
//              ptr += 1
//          new SparseTensor(cooIndices, cooValues, size, nnz, SparseLayout.COO)
//        case SparseLayout.CSC =>
//          val numCols = size(1)
//          val cooIndices = Tensor.zeros[Long](2, nnz)
//          val cooValues = Tensor.zeros[T](nnz)
//          val indptr = indices.select(0, *)
//          val cscIndices = indices.select(1, *)
//          var ptr = 0
//          for (i <- 0 until numCols)
//            val start = indptr(i).toInt
//            val end = indptr(i + 1).toInt
//            for (j <- start until end)
//              cooIndices(0, ptr) = cscIndices(j)
//              cooIndices(1, ptr) = i
//              cooValues(ptr) = values(j)
//              ptr += 1
//          new SparseTensor(cooIndices, cooValues, size, nnz, SparseLayout.COO)
//        case SparseLayout.BSR =>
//          // 这里可以实现从 BSR 到 COO 的转换逻辑
//          ???
//        case SparseLayout.BSC =>
//          // 这里可以实现从 BSC 到 COO 的转换逻辑
//          ???
//
//  // 转换为 CSC 格式
//  def to_spare_csc: SparseTensor[T] =
//    val numCols = size(1)
//    val cscIndptr = Tensor.zeros[Long](numCols + 1)
//    val cscIndices = Tensor.zeros[Long](nnz)
//    val cscValues = Tensor.zeros[T](nnz)
//    var ptr = 0
//    for (i <- 0 until numCols)
//      cscIndptr(i) = ptr
//      for (j <- 0 until nnz)
//        if (indices(1, j) == i)
//          cscIndices(ptr) = indices(0, j)
//          cscValues(ptr) = values(j)
//          ptr += 1
//    cscIndptr(numCols) = ptr
//    new SparseTensor(Tensor.stack(Seq(cscIndptr, cscIndices)), cscValues, size, nnz, SparseLayout.CSC)
//
//  // 转换为 BSR 格式
//  def to_spare_bsr: SparseTensor[T] =
//    val blockSize = (2, 2)
//    val numRows = size(0) / blockSize._1
//    val numCols = size(1) / blockSize._2
//    val bsrIndptr = Tensor.zeros[Long](numRows + 1)
//    val bsrIndices = Tensor.zeros[Long](nnz)
//    val bsrValues = Tensor.zeros[T](nnz)
//    var ptr = 0
//    for (i <- 0 until numRows)
//      bsrIndptr(i) = ptr
//      for (j <- 0 until nnz)
//        val blockRow = indices(0, j).toInt / blockSize._1
//        if (blockRow == i)
//          val blockCol = indices(1, j).toInt / blockSize._2
//          bsrIndices(ptr) = blockCol
//          bsrValues(ptr) = values(j)
//          ptr += 1
//    bsrIndptr(numRows) = ptr
//    new SparseTensor(Tensor.stack(Seq(bsrIndptr, bsrIndices)), bsrValues, size, nnz, SparseLayout.BSR)
//
//  // 转换为 BSC 格式
//  def to_sparse_bsc: SparseTensor[T] =
//    val blockSize = (2, 2)
//    val numRows = size(0) / blockSize._1
//    val numCols = size(1) / blockSize._2
//    val bscIndptr = Tensor.zeros[Long](numCols + 1)
//    val bscIndices = Tensor.zeros[Long](nnz)
//    val bscValues = Tensor.zeros[T](nnz)
//    var ptr = 0
//    for (i <- 0 until numCols)
//      bscIndptr(i) = ptr
//      for (j <- 0 until nnz)
//        val blockCol = indices(1, j).toInt / blockSize._2
//        if (blockCol == i)
//          val blockRow = indices(0, j).toInt / blockSize._1
//          bscIndices(ptr) = blockRow
//          bscValues(ptr) = values(j)
//          ptr += 1
//    bscIndptr(numCols) = ptr
//    new SparseTensor(Tensor.stack(Seq(bscIndptr, bscIndices)), bscValues, size, nnz, SparseLayout.BSC)
//
//  // 转换为密集张量
//  def to_Dense: Tensor[T] =
//    val denseTensor = Tensor.zeros[T](size: _*)
//    layout match
//      case SparseLayout.COO =>
//        for (i <- 0 until nnz)
//          val index = indices.select(1, i).toList.map(_.toInt)
//          denseTensor(index: _*) = values(i)
//      case SparseLayout.CSR =>
//        val coo = to_sparse_coo
//        for (i <- 0 until nnz)
//          val index = coo.indices.select(1, i).toList.map(_.toInt)
//          denseTensor(index: _*) = coo.values(i)
//      case SparseLayout.CSC =>
//        val coo = to_sparse_coo
//        for (i <- 0 until nnz)
//          val index = coo.indices.select(1, i).toList.map(_.toInt)
//          denseTensor(index: _*) = coo.values(i)
//      case SparseLayout.BSR =>
//        val coo = to_sparse_coo
//        for (i <- 0 until nnz)
//          val index = coo.indices.select(1, i).toList.map(_.toInt)
//          denseTensor(index: _*) = coo.values(i)
//      case SparseLayout.BSC =>
//        val coo = to_sparse_coo
//        for (i <- 0 until nnz)
//          val index = coo.indices.select(1, i).toList.map(_.toInt)
//          denseTensor(index: _*) = coo.values(i)
//    denseTensor
//
//  // 保持稀疏格式
//  def to_sparse: SparseTensor[T] = this
//
//
//object SparseTensor:
//  def apply[T](
//      indices: Tensor[Long],
//      values: Tensor[T],
//      size: Seq[Int],
//      nnz: Int,
//      layout: SparseLayout = SparseLayout.COO
//  ): SparseTensor[T] =
//    new SparseTensor(indices, values, size, nnz, layout)
