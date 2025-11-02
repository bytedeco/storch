package torch
package utils.data

import org.bytedeco.javacpp.{BoolPointer, SizeTPointer}
import org.bytedeco.javacpp.chrono.Milliseconds
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{FullDataLoaderOptions as FDLO, SizeTOptional, DataLoaderOptions as DO}

class DataOptions

class DataLoaderOptions(batchSize: Long) extends DO(batchSize) {

  override def batch_size(): SizeTPointer = super.batch_size()

  override def workers(): SizeTPointer = super.workers()

  override def max_jobs(): SizeTOptional = super.max_jobs()

  override def timeout(): Milliseconds = super.timeout()

  override def enforce_ordering(): BoolPointer = super.enforce_ordering()

  override def drop_last(): BoolPointer = super.drop_last()
}

class FullDataLoaderOptions(options: DataLoaderOptions) extends FDLO(options) {

  override def batch_size(): Long = super.batch_size()

  override def batch_size(setter: Long): FDLO = super.batch_size(setter)

  override def workers(): Long = super.workers()

  override def workers(setter: Long): FDLO = super.workers(setter)

  override def max_jobs(): Long = super.max_jobs()

  override def max_jobs(setter: Long): FDLO = super.max_jobs(setter)

  override def timeout(): Milliseconds = super.timeout()

  override def timeout(setter: Milliseconds): FDLO = super.timeout(setter)

  override def enforce_ordering(): Boolean = super.enforce_ordering()

  override def enforce_ordering(setter: Boolean): FDLO = super.enforce_ordering(setter)

  override def drop_last(): Boolean = super.drop_last()

  override def drop_last(setter: Boolean): FDLO = super.drop_last(setter)

}
