package cn.whu.edu.multiCoTraining;
import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.arbiter.data.DataSetIteratorProvider;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
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
import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * Created by bczhang on 2016/12/12.
 */
public class NNModel {
    public NNModel(String trainFileName,String testFileName) {
        this.trainFileName = trainFileName;
        this.testFileName=testFileName;
    }

    private DataSet trainingData;
    private DataSet testData;
    private String trainFileName;
    private String testFileName;
    DataSetIterator trainIter;
    DataSetIterator testIter;
    public List<Double> f1=new ArrayList<>();
    public List<Double>acc=new ArrayList<>();
    private  int classNum=3;
    //public MultiLayerNetwork GetNNModel(String trainFileName, List<Integer> addedDataSet) throws Exception {

    public MultiLayerNetwork GetNNModel(String trainFileName, List<DataSet> addedDataSet) throws Exception {

        int nEpochs = 150;
        int numOutputs = classNum;
        DataSet newTrainingDataSet=new DataSet();
//        if(addedDataSet!=null){
//            newTrainingDataSet=trainingData.copy();
//            for(DataSet d:trainingData){
//                addedDataSet.add(d);
//
//            }
//            newTrainingDataSet=newTrainingDataSet.merge(addedDataSet);
//
//            trainingData=newTrainingDataSet;
//            System.out.println("新的训练数据大小为"+trainingData.numExamples());
//        }
       // List<DataSet> traningD=trainingData.dataSetBatches(1);
        //Iterator<DataSet> trainIt=traningD.iterator();

        //得到网络配置文件
        MultiLayerConfiguration conf = this.LSTM_conf();
      //MultiLayerConfiguration conf = this.feedForward_conf();
        //model.setListeners(new ScoreIterationListener(100));  //Print score every 10 parameter updates
        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
            .epochTerminationConditions(new ScoreImprovementEpochTerminationCondition(5))
            .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES))
            .scoreCalculator(new DataSetLossCalculator(testIter, true))
            .evaluateEveryNEpochs(1)
            //.modelSaver(new LocalFileModelSaver("E:\\co-training\\model"))
            .build();
        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,conf,trainIter);
        //开始早停定型：
        EarlyStoppingResult result = trainer.fit();
      //获得最优模型：
        MultiLayerNetwork  bestModel=(MultiLayerNetwork)result.getBestModel();


          //MultiLayerNetwork bestModel = new MultiLayerNetwork(conf);
       // bestModel.init();
        // model.setListeners(new ScoreIterationListener(10));  //Print score every 10 parameter updates
//        for ( int n = 0; n < nEpochs; n++) {
//        while(trainIter.hasNext()) {
//            DataSet d=trainIter.next();
//            bestModel.fit(d);
//        }
//            trainIter.reset();
//           // trainIt=traningD.iterator();//list并没有提供，重置方法，可以重新生成一个
//       }
        //model评估
        System.out.println("1、得到"+trainFileName+"模型，并评估当前模型");
        Evaluation eval = new Evaluation(numOutputs);
        testIter.reset();
  while(testIter.hasNext()) {
      DataSet t = testIter.next();
      INDArray features = t.getFeatureMatrix();
      INDArray lables = t.getLabels();
      INDArray predicted = bestModel.output(features, false);
      eval.eval(lables, predicted);
      // System.out.println("训练数据集为："+trainFileName);
      //记录每次迭代结果，折线图
  }
        System.out.println(eval.f1());
        double   f   =   eval.f1();
        double   d   =   eval.accuracy();
        BigDecimal bd1   =   new   BigDecimal(f);
        BigDecimal bd2   =   new   BigDecimal(d);
        double   a   =   bd1.setScale(3,   BigDecimal.ROUND_HALF_UP).doubleValue();
        double   b   =   bd2.setScale(3,   BigDecimal.ROUND_HALF_UP).doubleValue();
        f1.add(a);
        acc.add(b);
        return bestModel;

    }

    public void init()throws  Exception{
        //每次迭代都要初始化，所以每次初始化都要清空当前状态空间
//        f1.clear();
//        acc.clear();
        int batchSize = 1;
        int batchSizetest = 1;
        //Load the training data:
        String localPath = "E:/co-training/sample/deeplearning4j/textLink/dblp/";
        //。。。。。。。。。。。初始化训练数据
        //得到文件长度用于初始化批大小
        String []f= FileUtils.readFileToString(new File(localPath+trainFileName+".csv")).trim().split("\n");
        System.out.println("初始化"+trainFileName+"训练数据："+f.length);
       // batchSize=f.length;
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(localPath+trainFileName+".csv")));
        trainIter = new RecordReaderDataSetIterator(rr,batchSize,0,classNum);
       // trainingData=trainIter.next();

        //。。。。。。。。。。。。初始化测试数据
        String []ftest= FileUtils.readFileToString(new File(localPath+testFileName+".csv")).trim().split("\n");
        System.out.println("初始化"+testFileName+"测试数据："+ftest.length);
      //  batchSizetest=ftest.length;
        RecordReader rrtest = new CSVRecordReader();
        rrtest.initialize(new FileSplit(new File(localPath+testFileName+".csv")));
        testIter = new RecordReaderDataSetIterator(rrtest,batchSizetest,0,classNum);
        testData=testIter.next();
        testIter.reset();
        //System.out.println("规范化训练数据和测试数据"+trainingData.numExamples());
        //DataNormalization normalizer = new NormalizerStandardize();
//        normalizer.fit(trainingData);
//        normalizer.transform(trainingData);
//        normalizer.transform(testData);

    }

    /**
     * LSTM 网络
     * @return a lstm network
     */
    private MultiLayerConfiguration LSTM_conf(){
        int inputNums=100;
        int middleHiddenNums=50;
        int outputNums=3;
        MultiLayerConfiguration LSTMconf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
            .updater(Updater.RMSPROP)
            .regularization(true).l2(1e-5)
            .weightInit(WeightInit.XAVIER)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
            .learningRate(0.0018)
            .list()
            .layer(0, new GravesLSTM.Builder().nIn(inputNums).nOut(middleHiddenNums)
                .activation("softsign").build())
            .layer(1, new RnnOutputLayer.Builder().activation("softmax")
                .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(middleHiddenNums).nOut(outputNums).build())
            .pretrain(false).backprop(true).build();
        return LSTMconf;
    }

    /**
     * 普通前馈网络
     * @param
     * @throws Exception
     */
    private MultiLayerConfiguration feedForward_conf(){
        int numInputs=100;
        int numHiddenNodes=50;
        int numOutputs=3;
        int seed = 123;
        double learningRate = 0.01;
        MultiLayerConfiguration ffconf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(1)//一次放入多少个训练，
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(learningRate)
            //.regularization(true).l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                .weightInit(WeightInit.XAVIER)
                .activation("tanh")
                .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .weightInit(WeightInit.XAVIER)
                .activation("softmax").weightInit(WeightInit.XAVIER)
                .nIn(numHiddenNodes).nOut(numOutputs).build())
            .pretrain(false).backprop(true).build();


        return ffconf;
    }
    public static void main(String[]args) throws  Exception{
        NNModel test=new NNModel("L_data1","Test_data1");
        test.init();
        test.GetNNModel("link", null);

    }
}
