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

object indexing:

  case class Slice(start: Option[Int], end: Option[Int], step: Option[Int])
  object Slice:
    private def extract(index: Option[Int] | Int) = index match
      case i: Option[Int] => i
      case i: Int         => Option(i)
    def apply(
        start: Option[Int] | Int = None,
        end: Option[Int] | Int = None,
        step: Option[Int] | Int = None
    ): Slice = Slice(extract(start), extract(end), extract(step))

  /** Ellipsis or ... in Python syntax. */
  sealed class Ellipsis

  /** Ellipsis or ... in Python syntax. */
  case object Ellipsis extends Ellipsis

  /** Ellipsis or ... in Python syntax. */
  val --- = Ellipsis

  /** Range (colon / :) in python syntax. */
  val :: = Slice()

  /** Allow for {{{t(1.::)}}} and {{{t(1.::(2)}}} */
  extension (start: Int | Option[Int])
    // python tensor[s:-1:t] [only have start ,step , end default -1 ,use as :-> s.::(t)
    def ::(step: Int | Option[Int]): Slice =
      // Note that despite the names, :: reverses the operators, that is a :: b calls b.::(a)
      // So step and start are reversed here
      Slice(step, None, start)
    // python tensor[0:-1:1]  use as ->  ::
    def :: : Slice = Slice(start, None, None)

    def by(step: Int | Option[Int]): Slice = Slice(start, None, step)

    // python tensor[s:e]  [only start end, step as default 1 ,use as  :-> s.&&(e)  slice(s,e,1)
  extension (start: Int | Option[Int])
    def &&(end: Int | Option[Int]): Slice =
      // Note that despite the names, :: reverses the operators, that is a :: b calls b.::(a)
      // So step and start are reversed here
      Slice(start, end, Some(1))

    def ::^(end: Int | Option[Int]): Slice =
      // Note that despite the names, :: reverses the operators, that is a :: b calls b.::(a)
      // So step and start are reversed here
      Slice(start, end, Some(1))
    def :&(end: Int | Option[Int]): Slice =
      // Note that despite the names, :: reverses the operators, that is a :: b calls b.::(a)
      // So step and start are reversed here
      Slice(start, end, Some(1))

    def untils(end: Int | Option[Int]): Slice =
      // Note that despite the names, :: reverses the operators, that is a :: b calls b.::(a)
      // So step and start are reversed here
      Slice(start, end, Some(1))

  // python tensor[s:e:t] [have start end step ], use as :-> s.&#(e,t)   slice(s,e,t)
  extension (start: Int | Option[Int])
    def &#(end: Int, step: Int): Slice = Slice(start, end, step)
    def :::*(end: Int, step: Int): Slice = Slice(start, end, step)

  // python tensor[ :e:t] [only end step] , use as : ->  e.&^(0.::(t))  -> slice(0,e,t)
  extension (end: Int | Option[Int])
    def &^(slice: Slice): Slice = Slice(slice.start, end, slice.step)
    def ^:^(slice: Slice): Slice = Slice(slice.start, end, slice.step)
    // Note that despite the names, :: reverses the operators, that is a :: b calls b.::(a)
    // So step and start are reversed here

  // python tensor[ :e:t] [only end step] , use as : ->  0.#&(e.::(t))  -> slice(0,e,t)
  extension (start: Int | Option[Int])
    def #&(slice: Slice): Slice = Slice(start, slice.start, slice.step)
    def :^^(slice: Slice): Slice = Slice(start, slice.start, slice.step)
    // Note that despite the names, :: reverses the operators, that is a :: b calls b.::(a)
    // So step and start are reversed here

//    def :: : Slice = Slice(start, None, None)

export indexing.*

/*
 *  python  scala
 *   :       ::  --- [all]
 *   k:      k.::    [only start]
 *   -k:      -k.::  [only start ]
 *   :k      0.&&(k)  [only end]
 *   ::k     0.::(k)  [only step]
 *   s:e:t    slice(s,e,t) 0.&#(e,t) [start, end, step]
 *   (k,r, c,..)  Seq(k,r,c..) [Select some index]
 *  s::k     s.::(k)  [only start and step]
 *  s:e       s.&&(e) [ only start end ]
 *  :e:t      slice(0,e,t) 0.#&(e.::(t)) e.&^(0.::(t)) [only end step]
 * */
