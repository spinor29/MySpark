import java.io._

import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.recommendation.{ALS, Rating, MatrixFactorizationModel}

object recommender {
    def main(args: Array[String]) {

        Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

        val conf = new SparkConf().setAppName("Recommender Application") // SparkConf object contains information about the application
        val sc = new SparkContext(conf) // initialize a SparkContext, pass a SparkConf object to the constructor

        // Load personal ratings
        val myRatings = sc.textFile("personalRatings.txt").map(_.split("::")).map { case Array(user, item, rate, time) => Rating(user.toInt, item.toInt, rate.toDouble)}
        println

        // Load movie ratings
        val data = sc.textFile("medium/ratings.dat")
        val ratings = data.map(_.split("::")).map { case Array(user, item, rate, time) => Rating(user.toInt, item.toInt, rate.toDouble)}

        // Randomly split the ratings RDD into three RDDs for training, validation, and test, respectively
        val Array(preTraining, validating, testing) = ratings.randomSplit(Array(0.6, 0.2, 0.2))
        val training = preTraining.union(myRatings)

        val movies = sc.textFile("medium/movies.dat").map { line => 
            val fields = line.split("::")
            (fields(0).toInt, fields(1))
        }.collect().toMap

        
        // Cross validation, find the best lambda
        val nf = 10 // number of features
        val lambdas = List(0.1, 1.0, 5.0, 10.0, 20.0) // regularization coefficient
        val nIter = 100 // number of iterations
        var bestModel: Option[MatrixFactorizationModel] = None
        var bestLambda = -1.0
        var minError = Double.MaxValue
        var output = "lambda, training error, validation error\n"
        for (lambda <- lambdas) {
            val model = ALS.train(training, nf, nIter, lambda)

            val error = computeError(model, validating)

            println("lambda = " + lambda)
            println("Error for validation set: " + error)

            val trainError = computeError(model, training)
            println("Error for training set: " + trainError)
            output += lambda + " " + trainError + " " + error + "\n"

            if (error < minError) {
                minError = error
                bestModel = Some(model)
                bestLambda = lambda
            }
        }

        val writer = new PrintWriter(new File("validationLog.txt"))
        writer.write(output)
        writer.close()
        
        println("Given nf = " + nf + " nIter = " + nIter + ",")
        println("The best model was obtained with lambda: " + bestLambda)
        val testError = computeError(bestModel.get, testing)
        println("Error for test set: " + testError)

        // Naive baseline
        val meanRating = training.map { case Rating(user, item, rate) => rate }.mean
        val baseError = math.sqrt(testing.map{ case Rating(user, item, rate) => (meanRating - rate) * (meanRating - rate) }.mean)

        val improvement = (baseError - testError) / baseError * 100.0

        println("\nThe best model improves the baseline by " + improvement + " %.")

        // Top recommendation
        val candidates = sc.parallelize(movies.keys.toSeq)
        val recommendations = bestModel.get.predict(candidates.map((0, _))).collect().sortBy(- _.rating).take(10)

        println("\nYour personal ratings:")
        myRatings.foreach {
            case Rating(user, item, rate) =>
                println(rate + " " + movies(item))
        }
        

        println("\nTop recommendations for you:")
        recommendations.foreach {
            case Rating(user, item, rate) =>
                println(rate + " " + movies(item))
        }
        
    }

    def computeError (model: MatrixFactorizationModel, data: RDD[Rating]): Double = {
        val userItems = data.map { case Rating(user, item, rate) => (user, item) }
        val predictions = model.predict(userItems)
        val ratesAndPreds = data.map { case Rating(user, item, rate) => ((user, item), rate) }
                            .join(predictions.map { case Rating(user, item, rate) => ((user, item), rate) })
                            .values
        val mse = ratesAndPreds.map(x => { val err = (x._1 - x._2); err * err }).reduce(_ + _) / data.count()
        //val mse = ratesAndPreds.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / data.count()
        math.sqrt(mse)
    }

}