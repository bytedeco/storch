import sbt._

import Keys.*
import MdocPlugin.autoImport.*
import LaikaPlugin.autoImport.*
import sbt.Def.settings
import sbtassembly.AssemblyPlugin.assemblySettings

ThisBuild / tlBaseVersion := "0.0" // your current series x.y

ThisBuild / organization := "io.github.mullerhai" //"dev.storch"
ThisBuild / organizationName := "storch.dev"
ThisBuild / startYear := Some(2022)
ThisBuild / licenses := Seq(License.Apache2)
ThisBuild / developers := List(
  // your GitHub handle and name
  tlGitHubDev("sbrunk", "Sören Brunk"),
  tlGitHubDev("mullerhai", "mullerhai")
)

// publish to s01.oss.sonatype.org (set to true to publish to oss.sonatype.org instead)
ThisBuild / tlSonatypeUseLegacyHost := false

// publish website from this branch
ThisBuild / tlSitePublishBranch := Some("main")

ThisBuild / apiURL := Some(new URL("https://storch.dev/api/"))

val scrImageVersion = "4.3.0"
val pytorchVersion = "2.7.1" // "2.5.1"
val cudaVersion =  "12.9-9.10" //"12.6-9.5"
val openblasVersion ="0.3.30" //"0.3.28"
val mklVersion = "2025.2"
ThisBuild / scalaVersion := "3.6.3"
ThisBuild / javaCppVersion := "1.5.12"  //"1.5.11"
ThisBuild / resolvers ++= Resolver.sonatypeOssRepos("snapshots")

ThisBuild / githubWorkflowJavaVersions := Seq(JavaSpec.temurin("11"))
ThisBuild / githubWorkflowOSes := Seq("macos-latest", "ubuntu-latest", "windows-latest")

val enableGPU = settingKey[Boolean]("enable or disable GPU support")

ThisBuild / enableGPU := false // true

val hasMKL = false
//val hasMKL = {
//  val firstPlatform = org.bytedeco.sbt.javacpp.Platform.current.head
//  firstPlatform == "linux-x86_64" || firstPlatform == "windows-x86_64"
//}

lazy val commonSettings = Seq(
  Compile / doc / scalacOptions ++= Seq("-groups","-Xno-duplicate-tasty","-Ycache-macro-class-loader:last-modified" ,"-snippet-compiler:compile","-Ywarn-conflicts","-Ywarn-unused","-Xno-enrich-error-messages","-Ycook-docs","-deprecation", "-feature","-Yresolve-term-conflict:package","--no-warnings"),
  javaCppVersion := (ThisBuild / javaCppVersion).value,
  javaCppPlatform := Seq(),
  // This is a hack to avoid depending on the native libs when publishing
  // but conveniently have them on the classpath during development.
  // There's probably a cleaner way to do this.
  tlJdkRelease := Some(11)
) ++ tlReplaceCommandAlias(
  "tlReleaseLocal",
  List(
    "reload",
    "project /",
    "set core / javaCppPlatform := Seq()",
    "set core / javaCppPresetLibs := Seq()",
    "+publishLocal"
  ).mkString("; ", "; ", "")
) ++ tlReplaceCommandAlias(
  "tlRelease",
  List(
    "reload",
    "project /",
    "set core / javaCppPlatform := Seq()",
    "set core / javaCppPresetLibs := Seq()",
    "+mimaReportBinaryIssues",
    "+publish",
    "tlSonatypeBundleReleaseIfRelevant"
  ).mkString("; ", "; ", "")
)
publishTo := sonatypePublishToBundle.value
// For all Sonatype accounts created on or after February 2021
//import xerial.sbt.Sonatype.sonatype01

//ThisBuild / sonatypeCredentialHost := sonatype01
import xerial.sbt.Sonatype.sonatypeCentralHost
ThisBuild / sonatypeCredentialHost := sonatypeCentralHost

import ReleaseTransformations._
releaseProcess := Seq[ReleaseStep](
  checkSnapshotDependencies,
  inquireVersions,
  runClean,
  runTest,
  setReleaseVersion,
  commitReleaseVersion,
  tagRelease,
  releaseStepCommandAndRemaining("+publishSigned"),
  releaseStepCommandAndRemaining("sonatypeBundleRelease"),
  setNextVersion,
  commitNextVersion,
  pushChanges
)
//libraryDependencies += "io.github.mullerhai" % "storch-scikit-learn_3" % "0.1.2-1.15.2" % Test exclude("org.scala-lang.modules","scala-collection-compat_2.13") exclude("org.typelevel","algebra_2.13")exclude("org.typelevel","cats-kernel_2.13")
// https://mvnrepository.com/artifact/org.bytedeco/cuda
//libraryDependencies += "org.bytedeco" % "pytorch-platform-gpu" % "2.7.1-1.5.12"
libraryDependencies += "org.bytedeco" % "cuda" % "12.9-9.10-1.5.12"
libraryDependencies += "org.apache.commons" % "commons-pool2" % "2.12.1"
// https://mvnrepository.com/artifact/com.fasterxml.jackson.core/jackson-databind
libraryDependencies += "com.fasterxml.jackson.core" % "jackson-databind" % "2.20.0"

//libraryDependencies += "ai.djl" % "api" % "0.33.0"
libraryDependencies += "com.alibaba.fastjson2" % "fastjson2" % "2.0.57"
//libraryDependencies += "org.bytedeco" % "cuda-platform" % "12.9-9.10-1.5.12"
libraryDependencies += "io.github.mullerhai" % "storch-numpy_3" % "0.1.7"
libraryDependencies += "io.github.mullerhai" % "storch-pickle_3" % "0.1.4"
libraryDependencies += "io.github.mullerhai" % "storch-tensorboard-proto_3" % "0.1.1"
libraryDependencies += "io.github.mullerhai" % "storch-plot_3" % "0.0.3"
libraryDependencies +=   "io.github.mullerhai" % "storch-scalapy_3" % "0.1.4-1.15.2" exclude("com.lihaoyi","sourcecode_2.13")
excludeDependencies += ExclusionRule(organization = "com.lihaoyi", name = "sourcecode_2.13")
excludeDependencies += ExclusionRule(organization = "spire", name = "compat_2.13")
excludeDependencies += ExclusionRule(organization = "spire", name = "compat_3")
excludeDependencies ++= Seq(
  "com.thesamet.scalapb" % "lenses_2.13" ,
  "com.thesamet.scalapb" % "scalapb-runtime_2.13",
//  "io.github.mullerhai" % "storch-safe-tensor_3",
//  "io.github.mullerhai" % "storch-polar_3",
  "com.lihaoyi" % "sourcecode_2.13",
  "spire" %"compat_2.13",
  "spire" %"compat_3"
)

lazy val storch_core = project
  .in(file("core"))
  .settings(commonSettings)
  .settings(
    javaCppPresetLibs ++= Seq(
      (if (enableGPU.value) "pytorch-gpu" else "pytorch") -> pytorchVersion,
      "openblas" -> openblasVersion
    ) ++ (if (hasMKL) Seq("mkl" -> mklVersion) else Seq()),
//    ++(if (enableGPU.value) Seq("cuda-redist" -> cudaVersion) else Seq()),
    javaCppPlatform := org.bytedeco.sbt.javacpp.Platform.current,
    fork := true,
    Test / fork := true,
    libraryDependencies ++= Seq(
      "org.bytedeco" % "pytorch" % s"$pytorchVersion-${javaCppVersion.value}",
//      "org.bytedeco" % "pytorch-platform-gpu" % s"$pytorchVersion-${javaCppVersion.value}",
      "org.bytedeco" % "cuda" % "12.9-9.10-1.5.12",
//      "org.bytedeco" % "cuda-platform" % "12.9-9.10-1.5.12",
      "org.typelevel" %% "spire" % "0.18.0",
      "org.typelevel" %% "shapeless3-typeable" % "3.3.0",
      "com.lihaoyi" %% "os-lib" % "0.9.1",
      "com.lihaoyi" %% "sourcecode" % "0.3.0",
      "dev.dirs" % "directories" % "26",
//      "ai.djl" % "api" % "0.33.0",
      "com.fasterxml.jackson.core" % "jackson-databind" % "2.20.0",
      "com.alibaba.fastjson2" % "fastjson2" % "2.0.57",
      "org.apache.commons" % "commons-pool2" % "2.12.1",
      "io.github.mullerhai" % "storch-numpy_3" % "0.1.7",
//      "io.github.mullerhai" % "storch-pandas_3" % "0.1.5",
      "io.github.mullerhai" % "storch-pickle_3" % "0.1.4",
      "io.github.mullerhai" % "storch-tensorboard-proto_3" % "0.1.1",
      "io.github.mullerhai" % "storch-plot_3" % "0.0.3",
      "io.github.mullerhai" % "storch-scalapy_3" % "0.1.4-1.15.2" exclude("com.lihaoyi","sourcecode_2.13"),
//      "io.github.mullerhai" % "storch-scikit-learn_3" % "0.1.2-1.15.2"  exclude("org.scala-lang.modules","scala-collection-compat_2.13") exclude("org.typelevel","algebra_2.13")exclude("org.typelevel","cats-kernel_2.13"),
      "org.scalameta" %% "munit" % "0.7.29" % Test,
      "org.scalameta" %% "munit-scalacheck" % "0.7.29" % Test
    ),
    excludeDependencies ++= Seq(
      "com.thesamet.scalapb" % "lenses_2.13" ,
      "com.lihaoyi" % "sourcecode_2.13",
      "com.thesamet.scalapb" % "scalapb-runtime_2.13",
//      "io.github.mullerhai" % "storch-safe-tensor_3",
//      "io.github.mullerhai" % "storch-polar_3",
      "spire" %"compat_2.13",
      "spire" %"compat_3"
    ),
    javaOptions ++= Seq(
      "-XX:+IgnoreUnrecognizedVMOptions",
      "--enable-native-access=ALL-UNNAMED",
      "--add-modules=jdk.incubator.vector",
      "--add-opens=java.base/java.lang=ALL-UNNAMED",
      "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
      "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
      "--add-opens=java.base/java.io=ALL-UNNAMED",
      "--add-opens=java.base/java.net=ALL-UNNAMED",
      "--add-opens=java.base/java.nio=ALL-UNNAMED",
      "--add-opens=java.base/java.util=ALL-UNNAMED",
      "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
      "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
      "--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED",
      "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
      "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
      "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
      "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
      "-Djdk.reflect.useDirectMethodHandle=false",
      "-Dio.netty.tryReflectionSetAccessible=true"
    )
  )

lazy val storch_vision = project
  .in(file("storch_vision"))
  .settings(commonSettings)
  .settings(
    libraryDependencies ++= Seq(
      "com.sksamuel.scrimage" % "scrimage-core" % scrImageVersion,
      "com.sksamuel.scrimage" % "scrimage-webp" % scrImageVersion,
      ("com.sksamuel.scrimage" %% "scrimage-scala" % scrImageVersion)
        .cross(CrossVersion.for3Use2_13),
      "org.scalameta" %% "munit" % "0.7.29" % Test
    ),
    excludeDependencies ++= Seq(
      "com.lihaoyi" % "sourcecode_2.13",
      "com.thesamet.scalapb" % "lenses_2.13",
      "com.thesamet.scalapb" % "scalapb-runtime_2.13"
    )
  )
  .dependsOn(storch_core)

lazy val storch_examples = project
  .in(file("storch_examples"))
  .enablePlugins(NoPublishPlugin)
  .settings(
    commonSettings,
    // disable discarded non-Unit value warnings in examples for now
    scalacOptions ~= (_.filterNot(Set("-Wvalue-discard")))
  )
  .settings(
    fork := true,
    libraryDependencies ++= Seq(
      "me.tongfei" % "progressbar" % "0.9.5",
      "com.github.alexarchambault" %% "case-app" % "2.1.0-M24",
      "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4"
    ),
    excludeDependencies ++= Seq(
      "com.lihaoyi" % "sourcecode_2.13",
      "com.thesamet.scalapb" % "lenses_2.13",
      "com.thesamet.scalapb" % "scalapb-runtime_2.13"
    )
  )
  .dependsOn(storch_vision)

lazy val docs = project
  .in(file("site"))
  .enablePlugins(ScalaUnidocPlugin, TypelevelSitePlugin, StorchSitePlugin)
  .settings(commonSettings)
  .settings(
    mdocVariables ++= Map(
      "JAVACPP_VERSION" -> javaCppVersion.value,
      "PYTORCH_VERSION" -> pytorchVersion,
      "OPENBLAS_VERSION" -> openblasVersion,
      "MKL_VERSION" -> mklVersion,
      "CUDA_VERSION" -> cudaVersion
    ),
    excludeDependencies ++= Seq(
      "com.lihaoyi" % "sourcecode_2.13",
      "org.scalameta" % "scalameta_2.13",
      "org.scalameta" % "parsers_2.13",
      "com.thesamet.scalapb" % "lenses_2.13",
      "com.thesamet.scalapb" % "scalapb-runtime_2.13"
    ),
    ScalaUnidoc / unidoc / unidocProjectFilter := inAnyProject -- inProjects(storch_examples),
    Laika / sourceDirectories ++= Seq(sourceDirectory.value),
    laikaIncludeAPI := true,
    laikaGenerateAPI / mappings := (ScalaUnidoc / packageDoc / mappings).value
  )
  .dependsOn(storch_vision)

lazy val storch_root = project
  .enablePlugins(NoPublishPlugin)
  .in(file("."))
  .aggregate(storch_core, storch_vision, storch_examples, docs)
  .settings(
    javaCppVersion := (ThisBuild / javaCppVersion).value,
    excludeDependencies ++= Seq(
      "com.thesamet.scalapb" % "lenses_2.13",
      "com.thesamet.scalapb" % "scalapb-runtime_2.13"
    )
  )

ThisBuild / assemblyMergeStrategy := {
  case v if v.contains("main$package$.class")                => MergeStrategy.first
  case v if v.contains("main.class")                         => MergeStrategy.first
  case v if v.contains("main$package.class")                 => MergeStrategy.first
  case v if v.contains("main$package.tasty")                 => MergeStrategy.first
  case v if v.contains("main.tasty")                         => MergeStrategy.first
  case v if v.contains("PointwiseOps.tasty")                 => MergeStrategy.first
  case v if v.contains("BLASOps.tasty")                      => MergeStrategy.first
  case v if v.contains("*.tasty")                            => MergeStrategy.first
  case v if v.contains("main.class")                         => MergeStrategy.discard
  case v if v.contains("module-info.class")                  => MergeStrategy.discard
  case v if v.contains("UnusedStub")                         => MergeStrategy.first
  case v if v.contains("aopalliance")                        => MergeStrategy.first
  case v if v.contains("inject")                             => MergeStrategy.first
  case v if v.contains("jline")                              => MergeStrategy.discard
  case v if v.contains("scala-asm")                          => MergeStrategy.discard
  case v if v.contains("asm")                                => MergeStrategy.discard
  case v if v.contains("scala-compiler")                     => MergeStrategy.deduplicate
  case v if v.contains("reflect-config.json")                => MergeStrategy.discard
  case v if v.contains("jni-config.json")                    => MergeStrategy.discard
  case v if v.contains("git.properties")                     => MergeStrategy.discard
  case v if v.contains("reflect.properties")                 => MergeStrategy.discard
  case v if v.contains("compiler.properties")                => MergeStrategy.discard
  case v if v.contains("scala-collection-compat.properties") => MergeStrategy.discard
  case x =>
    val oldStrategy = (assembly / assemblyMergeStrategy).value
    oldStrategy(x)
}
