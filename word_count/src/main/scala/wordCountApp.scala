import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object wordCountApp {
	def main(args: Array[String]) {
        val filename = "hamlet.txt"
        val conf = new SparkConf().setAppName("Word Count Application") // SparkConf object contains information about the application
        val sc = new SparkContext(conf) // initialize a SparkContext, pass a SparkConf object to the constructor
        val lines = sc.textFile(filename) // read a text file as a collection of lines
        // val lines = sc.textFile(filename).cache() // pull the data set into a cluster-wide in-memory cache
        val words = lines.flatMap(line => line.split(" ")) // split each line into words
        val pairs = words.map(word => (word, 1)) // map each word to a (word, 1) pair
        val counts = pairs.reduceByKey(_ + _) // or pairs.reduceByKey((a, b) => a + b); reduce value by key
        counts.saveAsTextFile("wc_out.txt")
	}
}
