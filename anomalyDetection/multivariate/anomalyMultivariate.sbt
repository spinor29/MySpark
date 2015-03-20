name := "Anomaly Detection Multivariate"

version := "1.0"

scalaVersion := "2.10.4"

libraryDependencies ++= Seq(
"org.apache.spark" %% "spark-core" % "1.3.0",
"org.apache.spark" %% "spark-mllib" % "1.3.0",
"com.github.fommil.netlib" % "all" % "1.1.2"
)
