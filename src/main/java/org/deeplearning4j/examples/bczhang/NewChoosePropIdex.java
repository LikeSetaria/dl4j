package org.deeplearning4j.examples.bczhang;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.*;

/**选择置信度高的K个样本
 * Created by bczhang on 2016/11/27.
 */
public class NewChoosePropIdex {
    public NewChoosePropIdex(String u_data_name,String another_U_dataName) {
        this.another_u_data_name=another_U_dataName;
        this.u_data_name=u_data_name;
    }

    private String another_u_data_name;
    private String u_data_name;
    private Set<String> handledUnlableSampleSet=new HashSet<>();
    private List<DataSet> U_datasetList=new LinkedList<>() ;
    private List<DataSet> anotherU_datasetList=new LinkedList<>() ;
    /**
     *
     * @param model
     * @param k
     * @return
     * @throws Exception
     */
    public List<DataSet> getKPropIndex(MultiLayerNetwork model,int k) throws  Exception{

        Map<Integer,Double> predictMap=new LinkedHashMap<>();
        Map<Integer,Double> KPropLineNum=new LinkedHashMap();
        INDArray PositiveLable = Nd4j.create(new float[]{1, 0}, new int[]{1, 2});
        INDArray negativeLable = Nd4j.create(new float[]{0, 1}, new int[]{1, 2});
        int numOutputs = 2;

        DataSet predictedDataset = new DataSet();
        predictedDataset= predictedDataset.merge(U_datasetList);
        //规范化
      //  DataNormalization normalizer = new NormalizerStandardize();
      //  normalizer.fit(predictedDataset);
       // normalizer.transform(predictedDataset);

        INDArray testPredicted = model.output(predictedDataset.getFeatures());
        System.out.println("结果行数为："+testPredicted.rows());
        INDArray  preRest= testPredicted.getColumn(0);
        for(int i=0;i<preRest.length();i++){
           // System.out.print(preRest.getDouble(i)+"   ");
            predictMap.put(i,preRest.getDouble(i));

        }
        predictMap=sortMap(predictMap,"desc");//对预测的结果进行排序
        KPropLineNum= getKProp(predictMap,k);//取置信度高的K个样本,这里得到的是行号，U_data1和U_data2具有一致的行号，所以根据这个去另一个U_data中取样本，so-called co-training
        DataSet u_dataSet=new DataSet();//
        List<DataSet> selectedDataSet=new LinkedList<>();

        INDArray features=null;
        INDArray labels= null;
        int counter=0,lineNum=0,num=0;
        for(DataSet d:anotherU_datasetList){
            if(KPropLineNum.containsKey(lineNum)) {

                DataSet dataSet = new DataSet();
                features = d.getFeatureMatrix();
                //这是对无标签数据进行选择，所以不可以直接取标签使用,不可以d.getLabels()
                if(KPropLineNum.get(lineNum)>0.5){
                    System.out.println(KPropLineNum.get(lineNum));
                labels =PositiveLable;//wenti
                    }
                else labels=negativeLable;
               // labels=d.getLabels();
                dataSet.setFeatures(features);
                dataSet.setLabels(labels);
                selectedDataSet.add(dataSet);
                //System.out.println("置信度高的K个样本所在行："+lineNum+"  ");
               // System.out.println(dataSet);

            }
           //System.out.println(d.getLabels()+"  " +d.getFeatures());
            lineNum++;
        }

        //删除已经选择的样本
        Map<Integer,Double> tempmap=sortMapbykey(KPropLineNum,"desc");
        for(Map.Entry<Integer,Double> d:tempmap.entrySet()){
            //System.out.println(anotherU_datasetList.size()+"删除index"+d.getKey());
            anotherU_datasetList.remove((int)d.getKey());
            U_datasetList.remove((int)d.getKey());

        }
     //   System.out.println("删除两个未标注数据集后，剩余的未标注的数据大小"+anotherU_datasetList.size()+"  "+U_datasetList.size());
        u_dataSet= u_dataSet.merge(selectedDataSet);
     //   System.out.println(u_dataSet);


        return selectedDataSet;

    }
   /**
   * 生成csv文件
   */
    public  void init()throws  Exception{
        String localPath = "E:\\co-training\\sample\\deeplearning4j\\";
        RecordReader rr = new CSVRecordReader();
        int batchSize = 1600;//设置批处理的数量，由于实验性数据较小，设置大数目
        //Load the test/evaluation data:

        RecordReader rrPridect = new CSVRecordReader();
        //rrTest.initialize(new FileSplit(new File(parentPath + "dl4j-examples/src/main/resources/classification/weibo_test_data.csv")));
        rrPridect.initialize(new FileSplit(new File(localPath +u_data_name+ ".csv")));//加载待预测的未标注的数据集
        DataSetIterator U_dataIter = new RecordReaderDataSetIterator(rrPridect, batchSize, 0, 2);

        RecordReader rrToSelect = new CSVRecordReader();
        //rrTest.initialize(new FileSplit(new File(parentPath + "dl4j-examples/src/main/resources/classification/weibo_test_data.csv")));
        rrToSelect.initialize(new FileSplit(new File(localPath +another_u_data_name+ ".csv")));//加载待预测的未标注的数据集
        DataSetIterator anotherU_dataIter = new RecordReaderDataSetIterator(rrToSelect, batchSize, 0, 2);
        //规范化
        DataSet UallData = U_dataIter.next();
        UallData.shuffle();

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(UallData);
        normalizer.transform(UallData);

        DataSet anotherUallData = anotherU_dataIter.next();
        anotherUallData.shuffle();

        DataNormalization normalizer2 = new NormalizerStandardize();
        normalizer.fit(anotherUallData);
        normalizer.transform(anotherUallData);

        for(DataSet d:UallData){
            U_datasetList.add(d);
        }
        for(DataSet d:anotherUallData){
            anotherU_datasetList.add(d);
        }
        System.out.println(u_data_name+"有dataset数据集:"+U_datasetList.size());
        System.out.println(another_u_data_name+"有dataset数据集:"+U_datasetList.size());
        System.out.println("初始化结束");

    }
    /**
     * 对map进行排序
     * @param oldMap
     * @param str 升序还是降序，只有是"desc"时是降序
     * @return
     */

    public static Map<Integer,Double> sortMap(Map<Integer,Double> oldMap,String str) {
        final String  order=str;//排序是升序还是降序，desc时是降序其它都默认为升序
        ArrayList<Map.Entry<Integer, Double>> list = new ArrayList<Map.Entry<Integer, Double>>(oldMap.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<Integer, Double>>() {

            @Override
            public int compare(Map.Entry<Integer, Double> arg0,
                               Map.Entry<Integer, Double> arg1) {
                double temp=-1;
                if(order.equals("desc")||order=="desc"){
               temp= arg0.getValue() - arg1.getValue();}
                else  {temp= arg1.getValue() - arg0.getValue();}
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
    /**
     * 对map进行排序
     * @param oldMap
     * @param str 升序还是降序，只有是"desc"时是降序
     * @return
     */

    public static Map<Integer,Double> sortMapbykey(Map<Integer,Double> oldMap,String str) {
        final String  order=str;//排序是升序还是降序，desc时是降序其它都默认为升序
        ArrayList<Map.Entry<Integer, Double>> list = new ArrayList<Map.Entry<Integer, Double>>(oldMap.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<Integer, Double>>() {

            @Override
            public int compare(Map.Entry<Integer, Double> arg0,
                               Map.Entry<Integer, Double> arg1) {
                int temp=-1;
                if(order.equals("desc")||order=="desc"){
                    temp= arg0.getKey() - arg1.getKey();}
                else  {temp= arg1.getKey() - arg0.getKey();}
                return -temp;
            }
        });
        Map newMap = new LinkedHashMap();
        for (int i = 0; i < list.size(); i++) {
            newMap.put(list.get(i).getKey(), list.get(i).getValue());
        }
        return newMap;
    }
    /**
     * 得到置信度比较高的K个样本，这K个中，K/2个来自置信度最高的，K/2来自置信度最低的
     */
    public static Map<Integer,Double> getKProp(Map<Integer,Double> sortedMap ,int K){
        if(K<0||sortedMap.size()<K){
            System.out.println("索引超出范围，请检查");
            return null;
        }
        int n,m;
        Map<Integer,Double> result=new LinkedHashMap<>();
        List<Integer> temp=new ArrayList<>();
        //判断K为奇数还是偶数
        if(K%2==0){
            n=K/2;
        }else {
            n=K/2+1;}
        m=K/2;

        for(Map.Entry<Integer,Double> entry: sortedMap.entrySet()){
            temp.add(entry.getKey());//把所有的key都加入到一个列表中
        }
        for(int i=0;i<n;i++){
            int key=temp.get(i);
            result.put(key,sortedMap.get(key));//不但要得到key即所在行，而且得到预测的概率
           // System.out.print(temp.get(i)+"  ");
        }
        for(int i=temp.size()-m;i<temp.size();i++){
            int key=temp.get(i);
            result.put(key,sortedMap.get(key));//不但要得到key即所在行，而且得到预测的概率
        }
        return result;
    }


}
