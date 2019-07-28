/*

object AirlinesSentiment {

}
*/
import org.apache.spark
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, RegexTokenizer, StopWordsRemover, StringIndexer, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.Row
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import java.io._

import scala.collection.mutable
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.functions._

object RunExample {

  def main(args: Array[String]): Unit= {
    val spark = org.apache.spark.sql.SparkSession.builder
      .master("local")
      .appName("Spark JSON Reader")
      .getOrCreate;
    val business=spark.read.option("multiLine", false).option("mode", "PERMISSIVE").json("business.json")
    //business.printSchema()
    //business.select("address","business_id","categories").show

    val review=spark.read.option("multiLine", false).option("mode", "PERMISSIVE").json("review.json")
    //review.printSchema()
    //review.select("business_id","date","text").show

    val user=spark.read.option("multiLine", false).option("mode", "PERMISSIVE").json("user.json")
    //user.printSchema()

    val tip=spark.read.option("multiLine", false).option("mode", "PERMISSIVE").json("tip.json")
//    tip.printSchema()
//
    val photo=spark.read.option("multiLine", false).option("mode", "PERMISSIVE").json("photo.json")
//    photo.printSchema()

    val check_in=spark.read.option("multiLine", false).option("mode", "PERMISSIVE").json("checkin.json")
    check_in.printSchema()











    /*

        //Create a SparkContext to initialize Spark
        val conf = new SparkConf()
        conf.setMaster("local")
        conf.setAppName("RunExample")
        val sc = new SparkContext(conf)
        val spark = org.apache.spark.sql.SparkSession.builder
          .master("local")
          .appName("Spark CSV Reader")
          .getOrCreate;

        val business = spark.read.option("inferSchema", "true").option("header", "true")
          .csv("business.json").toDF()

        val review = spark.read.option("inferSchema", "true").option("header", "true")
          .csv("review.json").toDF("reviewID","userID","businessID","stars","userful","funny","cool","text","date")


        review.select("userful","text").take(10).foreach(println)

    */
  }
}
