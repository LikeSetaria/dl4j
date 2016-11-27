package org.deeplearning4j.examples.bczhang;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.feedforward.classification.PlotUtil;
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
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

/**
 * Created by bczhang on 2016/11/26.
 */
public class TrainModel {


    public  MultiLayerNetwork GetNNModel(MultiLayerConfiguration conf) throws Exception {
       // int seed = 123;
       // double learningRate = 0.01;
        int batchSize = 50;
        int nEpochs = 30;

       // int numInputs = 12;
        int numOutputs = 2;
        //int numHiddenNodes = 350;

        //Load the training data:

        String parentPath="D:\\bczhang\\workspace\\ideaWorkplace\\dl4j-examples\\";
        RecordReader rr = new CSVRecordReader();
//        rr.initialize(new FileSplit(new File("src/main/resources/classification/linear_data_train.csv")));
        rr.initialize(new FileSplit(new File(parentPath+"dl4j-examples/src/main/resources/classification/weibo_train_data.csv")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,0,2);

        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(parentPath+"dl4j-examples/src/main/resources/classification/weibo_test_data.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,0,2);
//网络配置文件
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//            .seed(seed)
//            .iterations(1)
//            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//            .learningRate(learningRate)
//            .updater(Updater.NESTEROVS).momentum(0.9)
//            .list()
//            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
//                .weightInit(WeightInit.XAVIER)
//                .activation("relu")
//                .build())
//            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                .weightInit(WeightInit.XAVIER)
//                .activation("softmax").weightInit(WeightInit.XAVIER)
//                .nIn(numHiddenNodes).nOut(numOutputs).build())
//            .pretrain(false).backprop(true).build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));  //Print score every 10 parameter updates


        for ( int n = 0; n < nEpochs; n++) {
            model.fit( trainIter );
        }
//model评估
//        System.out.println("Evaluate model....");
//        Evaluation eval = new Evaluation(numOutputs);
//        while(testIter.hasNext()){
//            DataSet t = testIter.next();
//            INDArray features = t.getFeatureMatrix();
//            INDArray lables = t.getLabels();
//            INDArray predicted = model.output(features,false);
//
//            eval.eval(lables, predicted);
//
//        }

        //Print the evaluation statistics
       // System.out.println(eval.stats());


        //------------------------------------------------------------------------------------
        //Training is complete. Code that follows is for plotting the data & predictions only

        //Plot the data:
//        double xMin = 0;
//        double xMax = 1.0;
//        double yMin = -0.2;
//        double yMax = 0.8;
//
//        double xMin = -1;
//        double xMax = 100;
//        double yMin = 0.0;
//        double yMax = 1;
//
//        //Let's evaluate the predictions at every point in the x/y input space
//        int nPointsPerAxis = 500;
//        double[][] evalPoints = new double[nPointsPerAxis*nPointsPerAxis][12];
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

        System.out.println("****************Example finished********************");
        return model;
    }
}
