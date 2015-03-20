import scala.math._

import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}

object anomalyDetection {
    def main(args: Array[String]) {

        Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

        val conf = new SparkConf().setAppName("Anomaly Detection") // SparkConf object contains information about the application
        val sc = new SparkContext(conf) // initialize a SparkContext, pass a SparkConf object to the constructor

        val data_X = sc.textFile("X.dat")
        val X = data_X.map(_.split(',')).map { case Array(x1, x2) => Vectors.dense(x1.toDouble, x2.toDouble) }.cache()

        val data_Xval = sc.textFile("Xval.dat")
        val Xval = data_Xval.map(_.split(',')).map { case Array(x1, x2) => Vectors.dense(x1.toDouble, x2.toDouble) }

        val data_yval = sc.textFile("yval.dat")
        val yval = data_yval.map(_.split(',')).map { case Array(x) => x.toInt }.cache()

        //val prob = bivariate(X) // probability density
        
        def bivariate(X: RDD[Vector]): (Vector, Vector, Double) = {
            val summary: MultivariateStatisticalSummary = Statistics.colStats(X)
            val mu = summary.mean // a Vector

            val sigma2 = summary.variance
            //println(summary.variance)
            val correlMatrix: Matrix = Statistics.corr(X, "pearson")
            val rho = correlMatrix.toArray(1)

            (mu, sigma2, rho)
        }

        def density(X: RDD[Vector], mu: Vector, sigma2: Vector, rho: Double): RDD[Double] = {
            val sx = sqrt(sigma2(0))
            val sy = sqrt(sigma2(1))
            def f(v: Vector): Double = {
                val rx = (v(0)-mu(0))/sx
                val ry = (v(1)-mu(0))/sy
                1.0/(2.0*Pi*sx*sy*sqrt(1.0-rho*rho))*exp(-1.0/(2*(1.0-rho*rho))*(rx*rx + ry*ry - 2.0*rho*rx*ry))
            }
            X.map(f)
        }

        val stat = bivariate(X)


        println("\nAnomaly detection for server computers:")
        println("\nNumber of data points, [latency (ms), throughput(mb/s)]:")
        println(X.count())
        println("\nMean:")
        println(stat._1)
        println("Variance:")
        println(stat._2)
        println("Correlation coefficient:")
        println(stat._3)

        val prob = density(X, stat._1, stat._2, stat._3).cache()
        val pval = density(Xval, stat._1, stat._2, stat._3).cache()

        //val prob = bivariate(X).cache() // probability density
        //val pval = bivariate(Xval).cache()

        def selectThreshold(yval: RDD[Int], pval: RDD[Double]) = {
            val yWithIndex = yval.zipWithIndex.map{ case (s, i) => (i, s) }.cache()
            //var bestEpsilon = 0
            //var bestF1 = 0
            //var F1 = 0
            val stepsize = (pval.max - pval.min) / 1000.0

            //var epsilon = prob.min
            val max_pval = pval.max

            def best(epsi: Double, bestEpsilon: Double, bestf1: Double): (Double, Double) = {
                val predictions = pval.map( _ < epsi )
                // True positive
                val tp = predictions.zipWithIndex.map{ case(s, i) => (i, s) }
                         .join(yWithIndex)
                         .values
                         .filter( x => (x._1 == true) && (x._2 == 1) ).count().toDouble
                // False positive
                val fp = predictions.zipWithIndex.map{ case(s, i) => (i, s) }
                         .join(yWithIndex)
                         .values
                         .filter( x => (x._1 == true) && (x._2 == 0) ).count().toDouble
                //False negative
                val fn = predictions.zipWithIndex.map{ case(s, i) => (i, s) }
                         .join(yWithIndex)
                         .values
                         .filter( x => (x._1 == false) && (x._2 == 1)).count().toDouble
                val prec : Double = tp / (tp + fp)
                val rec : Double = tp / (tp + fn)
                val f1 : Double = 2.0 * prec * rec / (prec + rec)


                val bestSol = if (f1 > bestf1) (epsi, f1) else (bestEpsilon, bestf1)
                //println((epsi, f1, bestf1))


                if (epsi < max_pval) best(epsi + stepsize, bestSol._1, bestSol._2) else bestSol
            }

            best(pval.min, 0, 0)
        }

        // Selecting threshold
        println("\nFinding the best threshold (epsilon)...")
        val epsi_f1 = selectThreshold(yval, pval)

        print("\nThe best epsilon and F1 score:\n")
        println(epsi_f1)

        // Find outliers
        val outliers = prob.zipWithIndex.filter{ case (s,i) => (s < epsi_f1._1) }.values

        println
        println("Outlier points (index, x):")
        for (value <- outliers.collect()) {
            var i = value.toInt
            println((i,X.collect()(i)) + "\n")
        }

        println("Number of Outlier points: " + outliers.count())
    }

}
