package torch

package object distribute {

  def barrier[D <: DType](using backend: DistBackend[D]) = backend.barrier()

  def waitForPendingWorks[D <: DType](using backend: DistBackend[D]) = backend.waitForPendingWorks()

  def enableCollectivesTiming[D <: DType](using backend: DistBackend[D]) =
    backend.enableCollectivesTiming()

  def hasHooks[D <: DType](using backend: DistBackend[D]) = backend.hasHooks()

  def setGroupUid[D <: DType](uid: String)(using backend: DistBackend[D]) = backend.setGroupUid(uid)

  def getGroupUid[D <: DType](using backend: DistBackend[D]) = backend.getGroupUid()

  def getGroupDesc[D <: DType](using backend: DistBackend[D]) = backend.getGroupDesc()

  def getBoundDeviceId[D <: DType](using backend: DistBackend[D]) = backend.getBoundDeviceId()

  def reduce_scatter[D <: DType](tensorSeq: Seq[Tensor[D]], tensorSeq2: Seq[Tensor[D]])(using
      backend: DistBackend[D]
  ) = backend.reduce_scatter(tensorSeq, tensorSeq2)

  def all_reduce[D <: DType](tensors: Seq[Tensor[D]])(using backend: DistBackend[D]) =
    backend.allreduce(tensors)

  def allreduce_sparse[D <: DType](tensors: Seq[Tensor[D]])(using backend: DistBackend[D]) =
    backend.allreduce_sparse(tensors)

  def allreduce_coalesced[D <: DType](tensors: Seq[Tensor[D]])(using backend: DistBackend[D]) =
    backend.allreduce_coalesced(tensors)

  def supportsSplitting[D <: DType](using backend: DistBackend[D]) = backend.supportsSplitting()

  def startCoalescing[D <: DType](using backend: DistBackend[D]) = backend.startCoalescing()

  def endCoalescing[D <: DType](using backend: DistBackend[D]) = backend.endCoalescing()

  def getBackendName[D <: DType](using backend: DistBackend[D]) = backend.getBackendName()

  def get_gpu_id[D <: DType](using backend: DistBackend[D]) = backend.getID()

  def get_world_size[D <: DType](using backend: DistBackend[D]) = backend.getSize()

  def all_gather[D <: DType](
      tensorSeq: Seq[Tensor[D]],
      tensorSeq2: Seq[Tensor[D]],
      async_op: Boolean
  )(using backend: DistBackend[D]) = backend.allgather(tensorSeq, tensorSeq2, async_op)

  def reduce[D <: DType](tensors: Seq[Tensor[D]])(using backend: DistBackend[D]) =
    backend.reduce(tensors)

  def send[D <: DType](tensorSeq: Seq[Tensor[D]], dstRank: Int, tag: Int)(using
      backend: DistBackend[D]
  ) = backend.send(tensorSeq, dstRank, tag)

  def recv[D <: DType](tensorSeq: Seq[Tensor[D]], srcRank: Int, tag: Int)(using
      backend: DistBackend[D]
  ) = backend.recv(tensorSeq, srcRank, tag)

  def recvAnySource[D <: DType](tensorSeq: Seq[Tensor[D]], tag: Int)(using
      backend: DistBackend[D]
  ) = backend.recvAnysource(tensorSeq, tag)

  def allgather_coalesced[D <: DType](tensorSeq: Seq[Tensor[D]], tensorSeq2: Seq[Tensor[D]])(using
      backend: DistBackend[D]
  ) =
    backend.allgather_coalesced(tensorSeq, tensorSeq2)

  def allgather_into_tensor_coalesced[D <: DType](
      tensorSeq: Seq[Tensor[D]],
      tensorSeq2: Seq[Tensor[D]]
  )(using backend: DistBackend[D]) =
    backend.allgather_into_tensor_coalesced(tensorSeq, tensorSeq2)

  def gather[D <: DType](tensorSeq: Seq[Tensor[D]], tensorSeq2: Seq[Tensor[D]])(using
      backend: DistBackend[D]
  ) =
    backend.gather(tensorSeq, tensorSeq2)

  def scatter[D <: DType](tensorSeq: Seq[Tensor[D]], tensorSeq2: Seq[Tensor[D]])(using
      backend: DistBackend[D]
  ) =
    backend.scatter(tensorSeq, tensorSeq2)

  def reduce_scatter_tensor_coalesced[D <: DType](
      tensorSeq: Seq[Tensor[D]],
      tensorSeq2: Seq[Tensor[D]]
  )(using backend: DistBackend[D]) =
    backend.reduce_scatter_tensor_coalesced(tensorSeq, tensorSeq2)

  def all_to_all[D <: DType](tensorSeq: Seq[Tensor[D]], tensorSeq2: Seq[Tensor[D]])(using
      backend: DistBackend[D]
  ) = backend.alltoall(tensorSeq, tensorSeq2)

  def get_rank[D <: DType](using backend: DistBackend[D]) = backend.getRank()

  def broadcast[D <: DType](tensorSeq: Seq[Tensor[D]])(using backend: DistBackend[D]) =
    backend.broadcast(tensorSeq)
}
