package org.deeplearning4j.examples.bczhang;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.math.BigDecimal;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by bczhang on 2016/11/26.
 */
public class NewTrainModel {
    public static void main(String[]args)throws  Exception{
        NewTrainModel t=new NewTrainModel("U_data1");
        t.init();
        List<DataSet> g=null;
        t.GetNNModel("U_data1",  g);
    }

    public NewTrainModel(String trainFileName) {
        this.trainFileName = trainFileName;
    }
    private DataSet trainingData;
    private DataSet testData;
    private String trainFileName;
    public List<Double>f1=new ArrayList<>();
    public List<Double>acc=new ArrayList<>();


    public  MultiLayerNetwork GetNNModel(String trainFileName, List<DataSet> addedDataSet) throws Exception {
        int seed = 123;
        double learningRate = 0.01;

        int nEpochs = 50;
        int testBatchSize = 400;
        int numInputs = 12;
        int numOutputs = 2;
        int numHiddenNodes = 50;
        DataSet newTrainingDataSet=null;
         if(trainFileName.contains("L_data1"))
             numInputs=16;
          if(addedDataSet!=null){
              newTrainingDataSet=trainingData.copy();
              for(DataSet d:newTrainingDataSet.asList()){
                  addedDataSet.add(d);
              }
              newTrainingDataSet=newTrainingDataSet.merge(addedDataSet);
              trainingData=newTrainingDataSet;
             System.out.println("...."+newTrainingDataSet.numExamples());
             // trainingData=trainingData.merge(addedDataSet);

            //  System.out.println("合并进新的样本数为："+addedDataSet);

          }


       // System.out.println("新的总训练样本数为："+trainingData);

//网络配置文件
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(1)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(learningRate)
           // .regularization(true).l2(1e-4)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .weightInit(WeightInit.XAVIER)
                .activation("softmax").weightInit(WeightInit.XAVIER)
                .nIn(numHiddenNodes).nOut(numOutputs).build())
            .pretrain(false).backprop(true).build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
       // model.setListeners(new ScoreIterationListener(10));  //Print score every 10 parameter updates


        for ( int n = 0; n < nEpochs; n++) {
           // if( newTrainingDataSet!=null)
           // model.fit( newTrainingDataSet );
            //else
                model.fit(trainingData);

        }
//model评估
        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);

            INDArray features = testData.getFeatureMatrix();
            INDArray lables = testData.getLabels();
            INDArray predicted = model.output(features,false);
            eval.eval(lables, predicted);

        System.out.println("训练数据集为："+trainFileName);
        System.out.println(eval.stats());
        double   f   =   eval.f1();
        double   d   =   eval.accuracy();
        BigDecimal bd1   =   new   BigDecimal(f);
        BigDecimal bd2   =   new   BigDecimal(d);
        double   a   =   bd1.setScale(3,   BigDecimal.ROUND_HALF_UP).doubleValue();
        double   b   =   bd2.setScale(3,   BigDecimal.ROUND_HALF_UP).doubleValue();
        f1.add(a);
        acc.add(b);
        System.out.println("****************Example finished********************");
        return model;
    }
    public void init()throws  Exception{
        int batchSize = 50;
        //Load the training data:
        String localPath = "E:/co-training/sample/deeplearning4j/";
        //得到文件长度用于初始化批大小
        String []f= FileUtils.readFileToString(new File(localPath+trainFileName+".csv")).trim().split("\n");
        System.out.println(f.length);
        batchSize=f.length;
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(localPath+trainFileName+".csv")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,0,2);
        //对待处理的数据做一个规范化，因为数据量不大，这里是把所有的样本作为一个数据集，一次加载训练
        DataSet allData = trainIter.next();
        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.5);  //Use 65% of data for training

         trainingData = testAndTrain.getTrain();
         testData = testAndTrain.getTest();

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);
        normalizer.transform(trainingData);
        normalizer.transform(testData);
    }
}
