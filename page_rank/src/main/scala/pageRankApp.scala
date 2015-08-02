import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

class Page(val source: String, val links: Array[String]) 

object pageRankApp {
    def parsePage(line: String): Page = {
        val fields = line.split("[:,]")
        val source = fields(0)
        val links = fields.tail.map(_.trim())
        new Page(source, links)
    }

	def main(args: Array[String]) {
		val filename = args(0)
		val nIter = args(1).toInt // number of iterations
		val conf = new SparkConf().setAppName("Page Rank Application") // SparkConf object contains information about the application
        val sc = new SparkContext(conf) // initialize a SparkContext, pass a SparkConf object to the constructor
        val lines = sc.textFile(filename)
        val pages = lines.map(parsePage)
        val linksList = pages.map(p => (p.source, p.links))
        var ranks = pages.map(p => (p.source, 1.0))
        linksList.cache() // linksList does not change
        
        for (i <- 0 until nIter) {
            val contribs = linksList.join(ranks).flatMap {
            	case (source, (links, rank)) => links.map(l => (l, rank / links.size))
            } // contribs return an array of (l, rank / links.size)
            ranks = contribs.reduceByKey(_ + _).mapValues(0.15 + 0.85 * _) // equivalent to .map((k, v) => (k, 0.15 + 0.85 * v))
        }
        println("Final ranks:")
        ranks.collect().foreach(println)

        sc.stop()
	}
	
}
