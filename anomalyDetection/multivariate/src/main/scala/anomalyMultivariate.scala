import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian

object anomalyMultivariate {
    def main(args: Array[String]) {

        Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

        val conf = new SparkConf().setAppName("Anomaly Detection") // SparkConf object contains information about the application
        val sc = new SparkContext(conf) // initialize a SparkContext, pass a SparkConf object to the constructor

        val data_X = sc.textFile("X.dat")
        val X = data_X.map(_.split(',')).map { case x => x.map(_.toDouble) }.map(Vectors.dense).cache()

        val data_Xval = sc.textFile("Xval.dat")
        val Xval = data_Xval.map(_.split(',')).map { case x => x.map(_.toDouble) }.map(Vectors.dense).cache()

        val data_yval = sc.textFile("yval.dat")
        val yval = data_yval.map(_.split(',')).map { case Array(y) => y.toInt }.cache()

        def multiG(X: RDD[Vector]): (MultivariateGaussian, Vector, Vector) = {
            val summary: MultivariateStatisticalSummary = Statistics.colStats(X)
            val mu = summary.mean // a Vector
            val covX = (new RowMatrix(X)).computeCovariance() // a matrix

            // Covariance matrix should be symmetric. For some reason, the computeCovariance() of RowMatrix
            // does not give symmetric matrix; there is difference about 10^-14 in value between covX(i,j) and covX(j,i)
            // probably due to floating point error.
            val covX_sym = symmetrize(covX) // Make covX to be symmetric
            println(covX_sym)

            (new MultivariateGaussian(mu, covX_sym), mu, summary.variance)
        }
        
        def symmetrize(M: Matrix): Matrix = {
            val A_sym = new Array[Double](M.numCols * M.numRows)
            var i = 0
            for (i <- 0 until M.numCols) {
                var j = 0
                for (j <- 0 until M.numCols) {
                    A_sym(i*M.numCols + j) = if (i < j) M(j, i) else M(i, j)
                }
            }
            Matrices.dense(M.numRows, M.numCols, A_sym)
        }

        val gauss = multiG(X)
        val prob = X.map(gauss._1.pdf).cache()
        val pval = Xval.map(gauss._1.pdf).cache()
 
        println("\nAnomaly detection for server computers:")
        println("\nNumber of data points")
        println(X.count())
        println("\nMean:")
        println(gauss._2)
        println("Variance:")
        println(gauss._3)

        def selectThreshold(yval: RDD[Int], pval: RDD[Double]) = {
            val yWithIndex = yval.zipWithIndex.map{ case (s, i) => (i, s) }.cache()
            val stepsize = (pval.max - pval.min) / 1000.0

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
