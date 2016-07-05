package xu.xufly.sparkproject;

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */



// $example on$
import java.util.HashMap;
import java.util.Map;

import scala.Tuple2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.util.MLUtils;
// $example off$

class decisiontree {

  public static void main(String[] args) {
	    SparkConf sparkConf = new SparkConf().setAppName("JavaDecisionTreeClassificationExample").setMaster("local");
	    JavaSparkContext jsc = new JavaSparkContext(sparkConf);
	   String datapath = "/home/hadoop/input/libsvm.data";
	    JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), datapath).toJavaRDD();
    // $example on$
    
	    

    JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
    JavaRDD<LabeledPoint> trainingData = splits[0];
    JavaRDD<LabeledPoint> testData = splits[1];
    Integer numClasses = 2;
    Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
    String impurity = "entropy";
    Integer maxDepth = 30;
    Integer maxBins = 1000;
    final DecisionTreeModel model= DecisionTree.trainClassifier(trainingData, numClasses,
      categoricalFeaturesInfo, impurity, maxDepth, maxBins);
    JavaRDD<Tuple2<Object, Object>> predictionAndLabel=testData.map(new Function<LabeledPoint,Tuple2<Object,Object>>(){
    	    public  Tuple2<Object,Object> call(LabeledPoint p){
    	    	return new Tuple2<Object,Object>(model.predict(p.features()),p.label());
    	    }
    });

    // Get evaluation metrics.
    BinaryClassificationMetrics metrics2 = new BinaryClassificationMetrics(predictionAndLabel.rdd());
    
 // Get evaluation metrics.
    MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabel.rdd());
    
 // Precision by threshold
/*    JavaRDD<Tuple2<Object, Object>> precision = metrics2.precisionByThreshold().toJavaRDD();
    System.out.println("Precision by threshold: " + precision.toArray());*/
    
 // ROC Curve
    JavaRDD<Tuple2<Object, Object>> roc = metrics2.roc().toJavaRDD();
    System.out.println("ROC curve: " + roc.toArray());
    
 // AUROC
    System.out.println("Area under ROC = " + metrics2.areaUnderROC());
    
    
 // Confusion matrix
    Matrix confusion = metrics.confusionMatrix();
    System.out.println("Confusion matrix: \n" + confusion);
    
    
 // Overall statistics
    System.out.println("Precision = " + metrics.precision());
    System.out.println("Recall = " + metrics.recall());
    System.out.println("F1 Score = " + metrics.fMeasure());

    // Stats by labels
 /*   for (int i = 0; i < metrics.labels().length; i++) {
      System.out.format("Class %f precision = %f\n", metrics.labels()[i],metrics.precision
        (metrics.labels()[i]));
      System.out.format("Class %f recall = %f\n", metrics.labels()[i], metrics.recall(metrics
        .labels()[i]));
      System.out.format("Class %f F1 score = %f\n", metrics.labels()[i], metrics.fMeasure
        (metrics.labels()[i]));
    }*/

    //Weighted stats
/*    System.out.format("Weighted precision = %f\n", metrics.weightedPrecision());
    System.out.format("Weighted recall = %f\n", metrics.weightedRecall());
    System.out.format("Weighted F1 score = %f\n", metrics.weightedFMeasure());
    System.out.format("Weighted false positive rate = %f\n", metrics.weightedFalsePositiveRate());*/
    //System.out.println(predictionAndLabel.collect());
   /* Double testErr =1.0 * predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
        public Boolean call(Tuple2<Double, Double> pl) {
          return pl._1().equals(pl._2());
        }
      }).count() / testData.count();
*/
   // System.out.println("Test Error: " + testErr);
  //  System.out.println("Learned classification tree model:\n" + model.toDebugString());

    // Save and load model
  model.save(jsc.sc(), "home/hadoop/input/myDecisionTreeClassificationModel");
    DecisionTreeModel sameModel = DecisionTreeModel.load(jsc.sc(), "home/hadoop/input/myDecisionTreeClassificationModel");
    // $example off$
  }
}
