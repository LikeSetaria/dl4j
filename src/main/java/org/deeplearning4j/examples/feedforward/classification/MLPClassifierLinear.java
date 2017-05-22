package org.deeplearning4j.examples.feedforward.classification;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
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
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.concurrent.TimeUnit;

/**
 * "Linear" Data Classification Example
 *
 * Based on the data from Jason Baldridge:
 * https://github.com/jasonbaldridge/try-tf/tree/master/simdata
 *
 * @author Josh Patterson
 * @author Alex Black (added plots)
 *
 */
public class MLPClassifierLinear {


    public static void main(String[] args) throws Exception {
        int seed = 123;
        double learningRate = 0.01;
        int batchSize = 1;
        int nEpochs = 300;

        int numInputs = 100;
        int numOutputs = 3;
        int numHiddenNodes = 50;

        //Load the training data:

        String parentPath="D:\\bczhang\\workspace\\ideaWorkplace\\dl4j-examples\\";
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File( "E:/co-training/sample/deeplearning4j/textLink/dblp/L_data2.csv")));
       // rr.initialize(new FileSplit(new File(parentPath+"dl4j-examples/src/main/resources/classification/weibo_train_data.csv")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,0,3);

        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File("E:/co-training/sample/deeplearning4j/textLink/dblp/Test_data2.csv")));
        //rrTest.initialize(new FileSplit(new File(parentPath+"dl4j-examples/src/main/resources/classification/weibo_test_data.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,0,3);
//        DataNormalization normalizer = new NormalizerStandardize();
//        normalizer.fit(trainIter);

//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(seed)
//                .iterations(1)
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .learningRate(learningRate)
//                .updater(Updater.NESTEROVS).momentum(0.9)
//                .list()
//                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
//                        .weightInit(WeightInit.XAVIER)
//                        .activation("relu")
//                        .build())
//                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .weightInit(WeightInit.XAVIER)
//                        .activation("softmax").weightInit(WeightInit.XAVIER)
//                        .nIn(numHiddenNodes).nOut(numOutputs).build())
//                .pretrain(false).backprop(true).build();
//
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
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



        //model.setListeners(new ScoreIterationListener(100));  //Print score every 10 parameter updates
        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
            .epochTerminationConditions(new ScoreImprovementEpochTerminationCondition(5))
            .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES))
            .scoreCalculator(new DataSetLossCalculator(testIter, true))
            .evaluateEveryNEpochs(1)
            .modelSaver(new LocalFileModelSaver("E:\\co-training\\model"))
            .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,conf,trainIter);

//开始早停定型：
        EarlyStoppingResult result = trainer.fit();
//显示结果：
        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());

//获得最优模型：

        MultiLayerNetwork  bestModel=(MultiLayerNetwork)result.getBestModel();
     //System.out.println(bestModel.evaluate(testIter));
//        MultiLayerNetwork model = new MultiLayerNetwork(conf);
//        model.init();
//        for ( int n = 0; n < nEpochs; n++) {
//            model.fit( trainIter );
//        }

        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);
        testIter.reset();
        while(testIter.hasNext()){
            DataSet t = testIter.next();
            INDArray features = t.getFeatureMatrix();
            INDArray lables = t.getLabels();
            INDArray predicted = bestModel.output(features,false);

            eval.eval(lables, predicted);

        }

        //Print the evaluation statistics
        System.out.println(eval.stats());


        //------------------------------------------------------------------------------------
        //Training is complete. Code that follows is for plotting the data & predictions only

        //Plot the data:
//        double xMin = 0;
//        double xMax = 1.0;
//        double yMin = -0.2;
//        double yMax = 0.8;
        double xMin = -1;
        double xMax = 100;
        double yMin = 0.0;
        double yMax = 1;

        //Let's evaluate the predictions at every point in the x/y input space
//        int nPointsPerAxis = 500;
//        double[][] evalPoints = new double[nPointsPerAxis*nPointsPerAxis][100];
//        int count = 0;
//        for( int i=0; i<nPointsPerAxis; i++ ){
//            for( int j=0; j<nPointsPerAxis; j++ ){
//                double x = i * (xMax-xMin)/(nPointsPerAxis-1) + xMin;
//                double y = j * (yMax-yMin)/(nPointsPerAxis-1) + yMin;
//
//                evalPoints[count][0] = x;
//                evalPoints[count][1] = y;
//
//                count++;
//            }
//        }
//
//        INDArray allXYPoints = Nd4j.create(evalPoints);
//        INDArray predictionsAtXYPoints = model.output(allXYPoints);
//
//        //Get all of the training data in a single array, and plot it:
//        rr.initialize(new FileSplit(new File(parentPath+"dl4j-examples/src/main/resources/classification/weibo_train_data.csv")));
//        rr.reset();
//        int nTrainPoints = 1600;
//        trainIter = new RecordReaderDataSetIterator(rr,nTrainPoints,0,2);
//        DataSet ds = trainIter.next();
//        PlotUtil.plotTrainingData(ds.getFeatures(), ds.getLabels(), allXYPoints, predictionsAtXYPoints, nPointsPerAxis);
//
//
//        //Get test data, run the test data through the network to generate predictions, and plot those predictions:
//        rrTest.initialize(new FileSplit(new File(parentPath+"dl4j-examples/src/main/resources/classification/weibo_test_data.csv")));
//        rrTest.reset();
//        int nTestPoints = 400;
//        testIter = new RecordReaderDataSetIterator(rrTest,nTestPoints,0,2);
//        ds = testIter.next();
//        INDArray testPredicted = model.output(ds.getFeatures());
//        System.out.print(testPredicted);
//        PlotUtil.plotTestData(ds.getFeatures(), ds.getLabels(), testPredicted, allXYPoints, predictionsAtXYPoints, nPointsPerAxis);
//
//        System.out.println("****************Example finished********************");
    }
}
