package org.deeplearning4j.examples.bczhang;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.analysis.columns.IntegerAnalysis;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.*;

/**选择置信度高的K个样本
 * Created by bczhang on 2016/11/27.
 */
public class ChoosePropIdex {
    public ChoosePropIdex(String another_U_dataName) {
        this.another_u_data_name=another_U_dataName;
    }

    private static  List<DataSet> selectedTotalSamples=new LinkedList<>();//用以记录已经选择的sample，已经选择的co-training不再放入另一个中
    //调用的时候第一次是选择K个无标签的,第二次迭代就是2K,以此类推。。。
    private static Map<Integer,String> totalUnlabledSamples=new LinkedHashMap<>();
    private String another_u_data_name;
    private Set<String> handledUnlableSampleSet=new HashSet<>();

    /**
     *
     * @param model
     * @param k
     * @param U_dataName 待预测的无标签数据名
     * @param another_U_dataName 待选取的无标签数据名。根据U_data1预测的结果，去U_data2中选择K个加入到L_data2
     * @return
     * @throws Exception
     */
    public List<DataSet> getKPropIndex(MultiLayerNetwork model,int k,String U_dataName,String another_U_dataName) throws  Exception{

        Map<Integer,Double> predictMap=new LinkedHashMap<>();
        Map<Integer,Double> KPropLineNum=new LinkedHashMap();
        INDArray PositiveLable = Nd4j.create(new float[]{1, 0}, new int[]{1, 2});
        INDArray negativeLable = Nd4j.create(new float[]{0, 1}, new int[]{1, 2});
        int numOutputs = 2;
        //String parentPath = "D:\\bczhang\\workspace\\ideaWorkplace\\dl4j-examples\\";
        String localPath = "E:\\co-training\\sample\\deeplearning4j\\";
        RecordReader rr = new CSVRecordReader();
        int batchSize = 10000;//设置批处理的数量，由于实验性数据较小，设置大数目
        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        //rrTest.initialize(new FileSplit(new File(parentPath + "dl4j-examples/src/main/resources/classification/weibo_test_data.csv")));
        rrTest.initialize(new FileSplit(new File(localPath +U_dataName+ ".csv")));//加载待预测的未标注的数据集
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 2);

        RecordReader rrUnlableUdata = new CSVRecordReader();
        //Get test data, run the test data through the network to generate predictions, and plot those predictions:
        rrUnlableUdata.initialize(new FileSplit(new File(localPath + another_U_dataName+".csv")));
        DataSetIterator unlableIter = new RecordReaderDataSetIterator(rrUnlableUdata, batchSize, 0, 2);
        //rrTest.reset();

        DataSet ds = testIter.next();
        INDArray testPredicted = model.output(ds.getFeatures());
        System.out.println("结果行数为："+testPredicted.rows());

        INDArray  preRest= testPredicted.getColumn(0);
        for(int i=0;i<preRest.length();i++){
           // System.out.print(preRest.getDouble(i)+"   ");
            predictMap.put(i,preRest.getDouble(i));

        }
        predictMap=sortMap(predictMap,"ins");//对预测的结果进行排序
        KPropLineNum= getKProp(predictMap,k);//取置信度高的K个样本,这里得到的是行号，U_data1和U_data2具有一致的行号，所以根据这个去另一个U_data中取样本，so-called co-training
        DataSet u_dataSet=new DataSet();//
        List<DataSet> selectedDataSet=new LinkedList<>();
       // testIter.reset();//游标回到初始位置

        INDArray features=null;
        INDArray labels= null;
        int counter=0,lineNum=1,num=0;
        for(DataSet d:unlableIter.next()){
            if(KPropLineNum.containsKey(lineNum)) {
               // System.out.print("置信度高的K个样本所在行："+lineNum+"  ");
                DataSet dataSet = new DataSet();
                features = d.getFeatureMatrix();
                //这是对无标签数据进行选择，所以不可以直接取标签使用,不可以d.getLabels()
                if(KPropLineNum.get(lineNum)>0.5)
                labels =PositiveLable;//wenti
                else labels=negativeLable;
                dataSet.setFeatures(features);
                dataSet.setLabels(labels);
                selectedDataSet.add(dataSet);
                selectedTotalSamples.add(dataSet);
            }
         //   System.out.println(d.getLabels()+"  " +d.getFeatures());
            lineNum++;
        }
        u_dataSet= u_dataSet.merge(selectedDataSet);
       // System.out.println(u_dataSet.getLabels()+"  "+u_dataSet.getFeatures());
        /*
         *遍历获取K个追加到训练文件
//         */
//        StringBuilder strb=new StringBuilder();
//        for(Map.Entry<Integer,Double> entry:KPropLineNum.entrySet()){
//            String temp=totalUnlabledSamples.get(entry.getKey());
//            //System.out.println(temp);
//            if(!handledUnlableSampleSet.contains(temp)){
//            strb.append(temp);
//            strb.append("\n");}
//            handledUnlableSampleSet.add(temp);
//        }
//        if(another_U_dataName.equals("U_data1"))
//         FileUtils.write(new File("E:\\co-training\\sample\\deeplearning4j\\"+"L_data2.csv"),strb,true);
       // else if(another_U_dataName.equals("U_data2"))
         //   FileUtils.write(new File("E:\\co-training\\sample\\deeplearning4j\\"+"L_data1.csv"),strb,true);

        return selectedDataSet;

    }
   /**
   * 生成csv文件
   */
    public  void init()throws  Exception{
        String [] lines=FileUtils.readFileToString(new File("E:\\co-training\\sample\\deeplearning4j\\"+another_u_data_name+".csv")).trim().split("\n");

        int lineNum=1;
        for(String line:lines){
            totalUnlabledSamples.put(lineNum,line);
            //System.out.println(line);
            lineNum++;
        }
        System.out.println(another_u_data_name+"初始化结束");
    }
    /**
     * 对map进行排序
     * @param oldMap
     * @param str 升序还是降序，只有是"desc"时是降序
     * @return
     */

    public static Map sortMap(Map oldMap,String str) {
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
