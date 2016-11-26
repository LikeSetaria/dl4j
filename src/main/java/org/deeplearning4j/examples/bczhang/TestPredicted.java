package org.deeplearning4j.examples.bczhang;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.examples.feedforward.classification.PlotUtil;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

/**
 * Created by bczhang on 2016/11/26.
 */
public class TestPredicted {
    /**
     * 预测一个未标注的数据集的
     */
    public static void predicted(MultiLayerNetwork model) throws Exception{
        String parentPath="D:\\bczhang\\workspace\\ideaWorkplace\\dl4j-examples\\";
        RecordReader rr = new CSVRecordReader();
        int batchSize = 50;
        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(parentPath+"dl4j-examples/src/main/resources/classification/weibo_test_data.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,0,2);

        //Get test data, run the test data through the network to generate predictions, and plot those predictions:
        rrTest.initialize(new FileSplit(new File(parentPath+"dl4j-examples/src/main/resources/classification/weibo_test_data.csv")));
        rrTest.reset();

        DataSet  ds = testIter.next();
        INDArray testPredicted = model.output(ds.getFeatures());
        System.out.print(testPredicted);


        System.out.println("****************Example finished********************");
    }
}
