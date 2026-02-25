package torch.utils.gguf.impl

import torch.utils.gguf.{Builder, GGUF}

import java.io.IOException
import java.nio.channels.{ReadableByteChannel, WritableByteChannel}

object ImplAccessor {
  def newBuilder = new BuilderImpl

  def newBuilder(gguf: GGUF): Builder = BuilderImpl.fromExisting(gguf)

  def defaultAlignment: Int = ReaderImpl.ALIGNMENT_DEFAULT_VALUE

  def alignmentKey: String = ReaderImpl.ALIGNMENT_KEY

  @throws[IOException]
  def read(byteChannel: ReadableByteChannel): GGUF = new ReaderImpl().readImpl(byteChannel)

  @throws[IOException]
  def write(gguf: GGUF, byteChannel: WritableByteChannel): Unit = {
    WriterImpl.writeImpl(gguf, byteChannel)
  }
}
