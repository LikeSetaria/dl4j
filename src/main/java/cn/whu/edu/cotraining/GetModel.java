package cn.whu.edu.cotraining;

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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by bczhang on 2016/12/12.
 */
public class GetModel {
    public GetModel(String trainFileName,String testFileName) {
        this.trainFileName = trainFileName;
        this.testFileName=testFileName;
    }

    private DataSet trainingData;
    private DataSet testData;
    private String trainFileName;
    private String testFileName;
    public List<Double> f1=new ArrayList<>();
    public List<Double>acc=new ArrayList<>();


    public MultiLayerNetwork GetNNModel(String trainFileName, List<DataSet> addedDataSet) throws Exception {
        int seed = 123;
        double learningRate = 0.01;

        int nEpochs = 100;
        int testBatchSize = 400;
        int numInputs = 100;
        int numOutputs = 2;
        int numHiddenNodes = 50;
        DataSet newTrainingDataSet=null;
        if(addedDataSet!=null){
            newTrainingDataSet=trainingData.copy();
            for(DataSet d:newTrainingDataSet.asList()){
                addedDataSet.add(d);
            }
            newTrainingDataSet=newTrainingDataSet.merge(addedDataSet);
            trainingData=newTrainingDataSet;
            System.out.println("新的训练数据大小为"+trainingData.numExamples());
        }



//网络配置文件
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(1)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(learningRate)
            //.regularization(true).l2(1e-4)
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


//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//            .seed(seed)
//            .iterations(1)
//            .activation("relu")
//            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // use stochastic gradient descent as an optimization algorithm
//            .weightInit(WeightInit.XAVIER)
//            .learningRate(0.01)
//            .regularization(true).l2(1e-4)
//            .list()
//            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(70)
//                .build())
//            .layer(1, new DenseLayer.Builder().nIn(70).nOut(5)
//                .build())
//            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                .activation("softmax")
//                .nIn(5).nOut(numOutputs).build())
//            .backprop(true).pretrain(false)
//            .build();
//
//        //run the model
//        MultiLayerNetwork model = new MultiLayerNetwork(conf);
//        model.init();


        for ( int n = 0; n < nEpochs; n++) {
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
        //记录每次迭代结果，折线图
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
        int batchSizetest = 50;
        //Load the training data:
        String localPath = "E:/co-training/sample/deeplearning4j/textLink/dblp/";
        //。。。。。。。。。。。初始化训练数据
        //得到文件长度用于初始化批大小
        String []f= FileUtils.readFileToString(new File(localPath+trainFileName+".csv")).trim().split("\n");
        System.out.println("初始化"+trainFileName+"训练数据："+f.length);
        batchSize=f.length;
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(localPath+trainFileName+".csv")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,0,2);
        trainingData=trainIter.next();
        //对待处理的数据做一个规范化，因为数据量不大，这里是把所有的样本作为一个数据集，一次加载训练
        //。。。。。。。。。。。。初始化测试数据
        String []ftest= FileUtils.readFileToString(new File(localPath+testFileName+".csv")).trim().split("\n");
        System.out.println("初始化"+testFileName+"测试数据："+ftest.length);
        batchSizetest=ftest.length;
        RecordReader rrtest = new CSVRecordReader();
        rrtest.initialize(new FileSplit(new File(localPath+testFileName+".csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrtest,batchSizetest,0,2);
        testData=testIter.next();
//        DataSet allData = trainIter.next();  //每次从整个数据集中划分出来一个训练集测试集合
//       allData.shuffle();
//        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.4);  //Use 65% of data for training
//        trainingData = testAndTrain.getTrain();
//         testData = testAndTrain.getTest();
        //使用统一的规范化
//        List<DataSet>addedDataSet=testData.asList();
//        DataSet allData=null;
//        allData=trainingData.copy();
//        for(DataSet d:allData.asList()){
//            addedDataSet.add(d);
//        }
//        allData=allData.merge(addedDataSet);

        System.out.println("规范化训练数据和测试数据"+trainingData.numExamples());
        DataNormalization normalizer = new NormalizerStandardize();
//        normalizer.fit(trainingData);
//        normalizer.transform(trainingData);
//        normalizer.transform(testData);

    }
}
