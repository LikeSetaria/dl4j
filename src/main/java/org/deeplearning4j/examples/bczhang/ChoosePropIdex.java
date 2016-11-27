package org.deeplearning4j.examples.bczhang;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.analysis.columns.IntegerAnalysis;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.*;

/**选择置信度高的K个样本
 * Created by bczhang on 2016/11/27.
 */
public class ChoosePropIdex {
    public void getKPropIndex(MultiLayerNetwork model) throws  Exception{

        Map<Integer,Double> predictMap=new LinkedHashMap<>();


        int numOutputs = 2;
        String parentPath = "D:\\bczhang\\workspace\\ideaWorkplace\\dl4j-examples\\";
        RecordReader rr = new CSVRecordReader();
        int batchSize = 400;
        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(parentPath + "dl4j-examples/src/main/resources/classification/weibo_test_data.csv")));

        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 2);

        //Get test data, run the test data through the network to generate predictions, and plot those predictions:
        rrTest.initialize(new FileSplit(new File(parentPath + "dl4j-examples/src/main/resources/classification/weibo_test_data.csv")));
        rrTest.reset();

        DataSet ds = testIter.next();
        INDArray testPredicted = model.output(ds.getFeatures());
        System.out.println("结果行数为："+testPredicted.rows());

        INDArray  preRest= testPredicted.getColumn(0);
        for(int i=0;i<preRest.length();i++){
           // System.out.print(preRest.getDouble(i)+"   ");
            predictMap.put(i,preRest.getDouble(i));

        }
        predictMap=sortMap(predictMap);
        for(Map.Entry entry:predictMap.entrySet()){
            System.out.println(entry.getValue());
        }

    }


    public static Map sortMap(Map oldMap) {
        ArrayList<Map.Entry<Integer, Double>> list = new ArrayList<Map.Entry<Integer, Double>>(oldMap.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<Integer, Double>>() {

            @Override
            public int compare(Map.Entry<Integer, Double> arg0,
                               Map.Entry<Integer, Double> arg1) {
                double temp= arg0.getValue() - arg1.getValue();
                if(temp<0){
                return 1 ;}
                else if(temp>0) return -1;
                else return 0;
            }
        });
        Map newMap = new LinkedHashMap();
        for (int i = 0; i < list.size(); i++) {
            newMap.put(list.get(i).getKey(), list.get(i).getValue());
        }
        return newMap;
    }
}
