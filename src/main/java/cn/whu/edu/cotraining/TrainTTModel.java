package cn.whu.edu.cotraining;

import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Text-Link数据集co-training的实验
 * Created by bczhang on 2016/12/12.
 *
 */
public class TrainTTModel {
    private static Logger log = LoggerFactory.getLogger(TrainTTModel.class);

    private static Map<Integer,String> eats = readEnumCSV("/DataExamples/animals/eats.csv");
    private static Map<Integer,String> sounds = readEnumCSV("/DataExamples/animals/sounds.csv");
    private static Map<Integer,String> classifiers = readEnumCSV("/DataExamples/animals/classifiers.csv");
    public static void main(String[] args){

    try {

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        int labelIndex = 100;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
        int numClasses = 3;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2

        int batchSizeTraining =2000;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
        DataSet trainingData = readCSVDataset(
            // "/DataExamples/animals/animals_train.csv",
            "E:\\co-training\\sample\\deeplearning4j\\textLink\\test\\train.txt",
            batchSizeTraining, labelIndex, numClasses);

        // this is the data we want to classify
        int batchSizeTest = 400;
        DataSet testData = readCSVDataset("E:\\co-training\\sample\\deeplearning4j\\textLink\\test\\test.txt",
            batchSizeTest, labelIndex, numClasses);
        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(trainingData);     //Apply normalization to the training data
        normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set
        final int numInputs = 100;
        int outputNum = 3;
        int iterations = 20;
        long seed = 123;
        int nEpochs = 300;


        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .activation("relu")
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // use stochastic gradient descent as an optimization algorithm
            .weightInit(WeightInit.XAVIER)
            .learningRate(0.01)
            .regularization(true).l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(100)
                .build())
            .layer(1, new DenseLayer.Builder().nIn(100).nOut(20)
                .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation("softmax").weightInit(WeightInit.XAVIER)
                .nIn(20).nOut(outputNum).build())
            .backprop(true).pretrain(false)
            .build();


        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
      //  model.setListeners(new ScoreIterationListener(100));
        for ( int n = 0; n < nEpochs; n++) {
            model.fit(trainingData);
        }
        //evaluate the model on the test set
        Evaluation eval = new Evaluation(3);
        INDArray output = model.output(testData.getFeatureMatrix());
        eval.eval(testData.getLabels(), output);
        log.info(eval.stats());
        System.out.println(eval.stats());
    } catch (Exception e){
        e.printStackTrace();
    }

}

    public static Map<Integer,String> readEnumCSV(String csvFileClasspath) {
        try{
            List<String> lines = IOUtils.readLines(new ClassPathResource(csvFileClasspath).getInputStream());
            Map<Integer,String> enums = new HashMap<>();
            for(String line:lines){
                String[] parts = line.split(",");
                enums.put(Integer.parseInt(parts[0]),parts[1]);
            }
            return enums;
        } catch (Exception e){
            e.printStackTrace();
            return null;
        }

    }

    /**
     * used for testing and training
     *
     * @param csvFileClasspath
     * @param batchSize
     * @param labelIndex
     * @param numClasses
     * @return
     * @throws IOException
     * @throws InterruptedException
     */
    private static DataSet readCSVDataset(
        String csvFileClasspath, int batchSize, int labelIndex, int numClasses)
        throws IOException, InterruptedException{

        RecordReader rr = new CSVRecordReader();
        // rr.initialize(new FileSplit(new ClassPathResource(csvFileClasspath).getFile()));
        rr.initialize(new FileSplit(new File(csvFileClasspath)));
        DataSetIterator iterator = new RecordReaderDataSetIterator(rr,batchSize,labelIndex,numClasses);
        return iterator.next();
    }

}
