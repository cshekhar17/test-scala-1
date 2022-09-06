package org.anovos

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.{DataFrame, SQLContext}

import java.util

class CatToNumOHEScala(sqlContext: SQLContext, df: DataFrame) {
  import sqlContext.implicits._
  import org.apache.spark.sql.functions._
  import scala.collection.JavaConverters._

  def applyOHE(list_of_cols1 : util.ArrayList[String], list_of_cols_index1 : util.ArrayList[String], list_of_cols_vec1 : util.ArrayList[String]): DataFrame = {
    val list_of_cols = list_of_cols1.asScala.toArray
    val list_of_cols_index = list_of_cols_index1.asScala.toArray
    val list_of_cols_vec = list_of_cols_vec1.asScala.toArray

    val sampleIndexedDf = new StringIndexer().setHandleInvalid("skip").setInputCols(list_of_cols).setOutputCols(list_of_cols_index).fit(df).transform(df)
    val oneHotEncoder = new OneHotEncoder().setHandleInvalid("keep").setInputCols(list_of_cols_index).setOutputCols(list_of_cols_vec)
    val encoded = oneHotEncoder.fit(sampleIndexedDf).transform(sampleIndexedDf)

    val convertVectorToArr: Any => Array[Int] = _.asInstanceOf[SparseVector].toArray.map(_.toInt)
    val vectorToArrUdf = udf(convertVectorToArr)

    var odf = encoded
    odf = odf.drop(list_of_cols_index.toSeq:_*)
    val sample_row = odf.take(1)
    list_of_cols.foreach(colName => {
      val uniqCats = sample_row(0).getAs[SparseVector](colName+"_vec").size
      odf = odf.select(
        odf.col("*") +: (0 until uniqCats).map(i => vectorToArrUdf(col(colName+"_vec"))(i).alias(s"$colName-$i")): _*
      )
    })
    odf = odf.drop(list_of_cols_vec.toSeq:_*)
    odf
  }
}

