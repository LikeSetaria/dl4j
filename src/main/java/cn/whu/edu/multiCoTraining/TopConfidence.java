package cn.whu.edu.multiCoTraining;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import java.io.File;
import java.util.*;

/**
 * Created by bczhang on 2016/12/20.
 */
public class TopConfidence {
    public TopConfidence(String u_data_name,String another_U_dataName) {
        this.another_u_data_name=another_U_dataName;
        this.u_data_name=u_data_name;
       // this.classNum=classNum;
    }

    private String another_u_data_name;
    private String u_data_name;
    private List<DataSet> U_datasetList=new LinkedList<>() ;
    private List<DataSet> anotherU_datasetList=new LinkedList<>() ;
    /**
     *
     * @param model
     * @param k
     * @return
     * @throws Exception
     */
    public List<DataSet> getKPropIndex(MultiLayerNetwork model, int k) throws  Exception{

        Map<Integer,Double> predictMap=new LinkedHashMap<>();
        Map<Integer,Double> negativePredictMap=new LinkedHashMap<>();
        Map<Integer,Double> midPredictMap=new LinkedHashMap<>();
      ///  ArrayList<Map<Integer,Double>> predictedM=new ArrayList<>();
        Map<Integer,Double> KPropLineNum=new LinkedHashMap();
        Map<Integer,Double> negativeKPropLineNum=new LinkedHashMap();
        Map<Integer,Double> midKPropLineNum=new LinkedHashMap();
      ///  INDArray[] labels=new INDArray[classNum];
     ///   float[] f=new float[classNum];
    ///    int[] in=new int[classNum];
    ///    for(int i=0;i<classNum;i++){
     ///       in[i]=i+1;
     ///       f[i]=0;

     ///   }
     ///   for(int i=0;i<classNum;i++){
     ///       f[i]=1;
       ///     labels[i]=Nd4j.create(f, in);
      ///  }
        INDArray PositiveLable = Nd4j.create(new float[]{1,0,0});//对应类标签0
        INDArray negativeLable = Nd4j.create(new float[]{0,1,0});//对应类标签1
        INDArray midLable = Nd4j.create(new float[]{0,0,1});//对应类标签2

//System.out.println(PositiveLable+"   "+negativeLable+"  "+midLable);
        DataSet predictedDataset = new DataSet();
        predictedDataset= predictedDataset.merge(U_datasetList);

        INDArray testPredicted = model.output(predictedDataset.getFeatures());
        System.out.println("测试数据为："+U_datasetList.size());
      ///  INDArray[] predictedScores=new INDArray[classNum];
       /// for(int i=0;i<classNum;i++){
       ///     predictedScores[i]=testPredicted.getColumn(i);
      ///  }
        INDArray  preRest= testPredicted.getColumn(0);
        INDArray  negativePreRest= testPredicted.getColumn(1);
        INDArray midPreRest= testPredicted.getColumn(2);
      //  System.out.println(midPreRest);
        for(int i=0;i<preRest.length();i++){
            //System.out.print(preRest.getDouble(i)+"   ");

            predictMap.put(i,preRest.getDouble(i));
            negativePredictMap.put(i,negativePreRest.getDouble(i));
            midPredictMap.put(i,midPreRest.getDouble(i));

        }
        predictMap=sortMap(predictMap,"desc");//对预测的结果进行排序
        negativePredictMap=sortMap(negativePredictMap,"desc");//对预测的结果进行排序
        midPredictMap=sortMap(midPredictMap,"desc");//对预测的结果进行排序
        KPropLineNum= getKProp(predictMap,k);//取置信度高的K个样本,这里得到的是行号，U_data1和U_data2具有一致的行号，所以根据这个去另一个U_data中取样本，so-called co-training
        negativeKPropLineNum= getKProp(negativePredictMap,k);
        midKPropLineNum= getKProp(midPredictMap,k);
        System.out.println("类别一K个中最低置信度"+KPropLineNum);//类标号为2
        System.out.println("类别二K个中最低置信度"+negativeKPropLineNum);
        System.out.println("类别三K个中最低置信度"+midKPropLineNum);
        DataSet u_dataSet=new DataSet();//
        List<DataSet> selectedDataSet=new LinkedList<>();


        int counter=0,lineNum=0,num=0;
        for(DataSet d:anotherU_datasetList){
            INDArray features=null;
            INDArray labels= null;
            //System.out.println(lineNum+"  "+d.getLabels());
            if(KPropLineNum.containsKey(lineNum)) {
                DataSet dataSet = new DataSet();
                features = d.getFeatureMatrix();
                //labels =d.getLabels();
                labels =d.getLabels();
                //System.out.println(lineNum+"实际标签 "+d.getLabels()+" 预测概率"+KPropLineNum.get(lineNum)+"预测的标签"+labels);
                dataSet.setFeatures(features);
                dataSet.setLabels(PositiveLable);
                selectedDataSet.add(dataSet);
            }
            if(negativeKPropLineNum.containsKey(lineNum)) {
                DataSet dataSet = new DataSet();
                features = d.getFeatureMatrix();
                labels=d.getLabels();
                dataSet.setFeatures(features);
                dataSet.setLabels(negativeLable);
                selectedDataSet.add(dataSet);
            }
            if(midKPropLineNum.containsKey(lineNum)) {
                DataSet dataSet = new DataSet();
                features = d.getFeatureMatrix();
                labels=d.getLabels();
                dataSet.setFeatures(features);
                dataSet.setLabels(midLable);
                selectedDataSet.add(dataSet);
            }
            lineNum++;
        }
        //删除已经选择的样本
        KPropLineNum.putAll(negativeKPropLineNum);
        KPropLineNum.putAll(midKPropLineNum);
      //  System.out.println("删除的数目"+KPropLineNum.size());
        Map<Integer,Double> tempmap=sortMapbykey(KPropLineNum,"desc");
        for(Map.Entry<Integer,Double> d:tempmap.entrySet()){
            anotherU_datasetList.remove((int)d.getKey());
            U_datasetList.remove((int)d.getKey());
        }
      //  System.out.println("删除两个未标注数据集后，剩余的未标注的数据大小"+anotherU_datasetList.size()+"  "+U_datasetList.size());
        return selectedDataSet;
    }

    /**
     * 获取置信度高的K个样本的第二个实现
     * 选择两个分类器评分都高的K个，然后加入到两个分类器中
     */
    public Map<String,List<DataSet>> getKPropIndex(MultiLayerNetwork textModel,MultiLayerNetwork relationModel,int k){

        Map<Integer,Double> predictTextMap=new LinkedHashMap<>();
        Map<Integer,Double> negativePredictTextMap=new LinkedHashMap<>();
        Map<Integer,Double> midPredictTextMap=new LinkedHashMap<>();

        Map<Integer,Double> predictRelationMap=new LinkedHashMap<>();
        Map<Integer,Double> negativePredictRelationMap=new LinkedHashMap<>();
        Map<Integer,Double> midPredictRelationMap=new LinkedHashMap<>();

        Map<Integer,Double> KPropLineNum;
        Map<Integer,Double> negativeKPropLineNum;
        Map<Integer,Double> midKPropLineNum;
        Map<String,List<DataSet>> resutMap=new LinkedHashMap<>() ;
        INDArray PositiveLable = Nd4j.create(new float[]{1,0,0});
        INDArray negativeLable = Nd4j.create(new float[]{0,1,0});
        INDArray midLable = Nd4j.create(new float[]{0,0,1});

        DataSet predictedTextDataset = new DataSet();
        DataSet predictedRelationDataset = new DataSet();
        predictedTextDataset= predictedTextDataset.merge(U_datasetList);
        predictedRelationDataset= predictedRelationDataset.merge(anotherU_datasetList);


        INDArray testTextPredicted = textModel.output(predictedTextDataset.getFeatures());
        INDArray testRelationPredicted = relationModel.output(predictedRelationDataset.getFeatures());
        System.out.println("unlabled文本结果行数为："+testTextPredicted.rows());
        System.out.println("unlabled关系结果行数为："+testRelationPredicted.rows());
        INDArray  preTextRest= testTextPredicted.getColumn(0);
        INDArray  preTextRest2= testTextPredicted.getColumn(1);
        INDArray  preTextRest3= testTextPredicted.getColumn(2);
        INDArray  preRelationRest= testRelationPredicted.getColumn(0);
        INDArray  preRelationRest2= testRelationPredicted.getColumn(1);
        INDArray  preRelationRest3= testRelationPredicted.getColumn(2);
        for(int i=0;i<preTextRest.length();i++){
            predictTextMap.put(i,preTextRest.getDouble(i));
            negativePredictTextMap.put(i,preTextRest2.getDouble(i));
            midPredictTextMap.put(i,preTextRest3.getDouble(i));
        }
        for(int i=0;i<preRelationRest.length();i++){
            predictRelationMap.put(i,preRelationRest.getDouble(i));
            negativePredictRelationMap.put(i,preRelationRest2.getDouble(i));
            midPredictRelationMap.put(i,preRelationRest3.getDouble(i));
        }
        predictTextMap=sortMap(predictTextMap,"desc");//对预测的结果进行排序
        negativePredictTextMap=sortMap(negativePredictTextMap,"desc");//对预测的结果进行排序
        midPredictTextMap=sortMap(midPredictTextMap,"desc");//对预测的结果进行排序

        predictRelationMap=sortMap(predictRelationMap,"desc");//对预测的结果进行排序
        negativePredictRelationMap=sortMap(negativePredictRelationMap,"desc");//对预测的结果进行排序
        midPredictRelationMap=sortMap(midPredictRelationMap,"desc");//对预测的结果进行排序

        KPropLineNum= getKProp(predictTextMap,predictRelationMap,k);//取置信度高的K个样本,这里得到的是行号，U_data1和U_data2具有一致的行号，所以根据这个去另一个U_data中取样本，so-called co-training
        negativeKPropLineNum=getKProp(negativePredictTextMap,negativePredictRelationMap,k);
        midKPropLineNum=getKProp(midPredictTextMap,midPredictRelationMap,k);

         //System.out.println(KPropLineNum.size()+"  "+negativeKPropLineNum.size()+"    "+midKPropLineNum.size());
        DataSet u_dataSet=new DataSet();//
        List<DataSet> selectedTextDataSet=new LinkedList<>();
        List<DataSet> selectedRelationDataSet=new LinkedList<>();

        INDArray features=null;
        INDArray labels= null;
        int counter=0,lineNum=0;
        for(DataSet d:anotherU_datasetList){
            if(KPropLineNum.containsKey(lineNum)) {
                DataSet dataSet = new DataSet();
                features = d.getFeatureMatrix();
                //这是对无标签数据进行选择，所以不可以直接取标签使用,不可以d.getLabels()
                labels =PositiveLable;//wenti
                dataSet.setFeatures(features);
                dataSet.setLabels(labels);
                selectedTextDataSet.add(dataSet);
            }
            if(negativeKPropLineNum.containsKey(lineNum)) {
                DataSet dataSet = new DataSet();
                features = d.getFeatureMatrix();
                //这是对无标签数据进行选择，所以不可以直接取标签使用,不可以d.getLabels()
                labels=negativeLable;
                dataSet.setFeatures(features);
                dataSet.setLabels(labels);
                selectedTextDataSet.add(dataSet);
            }
            if(midKPropLineNum.containsKey(lineNum)) {
                DataSet dataSet = new DataSet();
                features = d.getFeatureMatrix();
                //这是对无标签数据进行选择，所以不可以直接取标签使用,不可以d.getLabels()
                labels=midLable;
                dataSet.setFeatures(features);
                dataSet.setLabels(labels);
                selectedTextDataSet.add(dataSet);
            }
            //System.out.println(d.getLabels()+"  " +d.getFeatures());
            lineNum++;
        }
        for(DataSet d:U_datasetList){
            if(KPropLineNum.containsKey(counter)) {
                DataSet dataSet = new DataSet();
                features = d.getFeatureMatrix();
                //这是对无标签数据进行选择，所以不可以直接取标签使用,不可以d.getLabels()
                labels =PositiveLable;//wenti
                dataSet.setFeatures(features);
                dataSet.setLabels(labels);
                selectedRelationDataSet.add(dataSet);
            }
            if(negativeKPropLineNum.containsKey(counter)) {
                DataSet dataSet = new DataSet();
                features = d.getFeatureMatrix();
                //这是对无标签数据进行选择，所以不可以直接取标签使用,不可以d.getLabels()
                labels=negativeLable;
                dataSet.setFeatures(features);
                dataSet.setLabels(labels);
                selectedRelationDataSet.add(dataSet);
            }
            if(midKPropLineNum.containsKey(counter)) {
                DataSet dataSet = new DataSet();
                features = d.getFeatureMatrix();
                //这是对无标签数据进行选择，所以不可以直接取标签使用,不可以d.getLabels()
                labels=midLable;
                dataSet.setFeatures(features);
                dataSet.setLabels(labels);
                selectedRelationDataSet.add(dataSet);
            }

            //System.out.println(d.getLabels()+"  " +d.getFeatures());
            counter++;
        }
       // System.out.println("加入的text  "+selectedTextDataSet.size()+"加入的link  "+selectedRelationDataSet.size());
        resutMap.put("text",selectedTextDataSet);
        resutMap.put("relation",selectedRelationDataSet);
        //删除已经选择的样本

        Set<Integer> allRemoveID=new LinkedHashSet<>();
        Map<Integer,Double> tempmap=sortMapbykey(KPropLineNum,"desc");
        Map<Integer,Double> tempmap2=sortMapbykey(negativeKPropLineNum,"desc");
        Map<Integer,Double> tempmap3=sortMapbykey(midKPropLineNum,"desc");
        for(Map.Entry<Integer,Double> d:tempmap.entrySet()){
            //System.out.println(anotherU_datasetList.size()+"删除index"+d.getKey());
            allRemoveID.add(d.getKey());
//            anotherU_datasetList.remove((int)d.getKey());
//            U_datasetList.remove((int)d.getKey());
        }
        for(Map.Entry<Integer,Double> d:tempmap2.entrySet()){
            //System.out.println(anotherU_datasetList.size()+"删除index"+d.getKey());
            allRemoveID.add(d.getKey());
//            anotherU_datasetList.remove((int)d.getKey());
//            U_datasetList.remove((int)d.getKey());
        }
        for(Map.Entry<Integer,Double> d:tempmap3.entrySet()){
            //System.out.println(anotherU_datasetList.size()+"删除index"+d.getKey());
            allRemoveID.add(d.getKey());
//            anotherU_datasetList.remove((int)d.getKey());
//            U_datasetList.remove((int)d.getKey());
        }
        for(int id:allRemoveID){
            if(anotherU_datasetList.contains(id))
            anotherU_datasetList.remove(id);
            if(U_datasetList.contains(id))
            U_datasetList.remove(id);
        }
        //   System.out.println("删除两个未标注数据集后，剩余的未标注的数据大小"+anotherU_datasetList.size()+"  "+U_datasetList.size());
        // u_dataSet= u_dataSet.merge(selectedDataSet);
        //   System.out.println(u_dataSet);
        return resutMap;

    }


    /**
     * 生成csv文件
     */
    public  void init()throws  Exception{
        String localPath = "E:\\co-training\\sample\\deeplearning4j\\textLink\\dblp\\";
        RecordReader rr = new CSVRecordReader();
        int batchSize = 6563;//设置批处理的数量，由于实验性数据较小，设置大数目
        //Load the test/evaluation data:
        String []f= FileUtils.readFileToString(new File(localPath+u_data_name+".csv")).trim().split("\n");
        System.out.println("初始化"+u_data_name+"训练数据："+f.length);
        batchSize=f.length;
        RecordReader rrPridect = new CSVRecordReader();
        //rrTest.initialize(new FileSplit(new File(parentPath + "dl4j-examples/src/main/resources/classification/weibo_test_data.csv")));
        rrPridect.initialize(new FileSplit(new File(localPath +u_data_name+ ".csv")));//加载待预测的未标注的数据集
        DataSetIterator U_dataIter = new RecordReaderDataSetIterator(rrPridect, batchSize, 0, 3);

        RecordReader rrToSelect = new CSVRecordReader();
        //rrTest.initialize(new FileSplit(new File(parentPath + "dl4j-examples/src/main/resources/classification/weibo_test_data.csv")));
        rrToSelect.initialize(new FileSplit(new File(localPath +another_u_data_name+ ".csv")));//加载待预测的未标注的数据集
        DataSetIterator anotherU_dataIter = new RecordReaderDataSetIterator(rrToSelect, batchSize, 0, 3);
        //规范化
        DataSet UallData = U_dataIter.next();
        //UallData.shuffle();

        DataNormalization normalizer = new NormalizerStandardize();
//        normalizer.fit(UallData);
//        normalizer.transform(UallData);

        DataSet anotherUallData = anotherU_dataIter.next();
        //anotherUallData.shuffle();

        DataNormalization normalizer2 = new NormalizerStandardize();
//        normalizer2.fit(anotherUallData);
//        normalizer2.transform(anotherUallData);

//int m=1;
        for(DataSet d:UallData){
            U_datasetList.add(d);
            //System.out.println(m+" 行号"+d.getFeatures()+"   "+d.getLabels());
            //m++;
        }
        for(DataSet d:anotherUallData){
            anotherU_datasetList.add(d);
        }
        System.out.println(u_data_name+"有dataset数据集:"+U_datasetList.size());
        System.out.println(another_u_data_name+"有dataset数据集:"+anotherU_datasetList.size());
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

    public static Map<DataSet,Double> newsortMap(Map<DataSet,Double> oldMap,String str) {
        final String  order=str;//排序是升序还是降序，desc时是降序其它都默认为升序
        ArrayList<Map.Entry<DataSet, Double>> list = new ArrayList<Map.Entry<DataSet, Double>>(oldMap.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<DataSet, Double>>() {

            @Override
            public int compare(Map.Entry<DataSet, Double> arg0,
                               Map.Entry<DataSet, Double> arg1) {
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
     * 重写getKProp，根据两个集合选择，选择置信度都比较高的交集的K个
     */
    public static Map<Integer,Double> getKProp(Map<Integer,Double> sortedTextMap ,Map<Integer,Double> sortedRelationMap,int K){
        if(K<0||sortedTextMap.size()<K){
            System.out.println("索引超出范围");
            return null;
        }
        int n,m;
        Map<Integer,Double> result=new LinkedHashMap<>();

        List<Integer> temp1=new ArrayList<>();
        List<Integer> temp2=new ArrayList<>();
        List<Integer> temp=new ArrayList<>();

        for(Map.Entry<Integer,Double> entry: sortedTextMap.entrySet()){
            temp1.add(entry.getKey());//把所有的key都加入到一个列表中
        }
        for(Map.Entry<Integer,Double> entry: sortedRelationMap.entrySet()){
            temp2.add(entry.getKey());//把所有的key都加入到一个列表中
        }
        int count=1;
        //取两个分类器评分都高的K个样本
//        for(int i=0;i<temp1.size()&&count<=K;i++){
//            for(int j=0;j<100;j++) {
//                if ((i+j)<temp1.size()&&temp1.get(i) == temp2.get(i+j)) {
//                    System.out.println(i + "次  " + temp1.get(i));
//                    temp.add(temp1.get(i));
//                    count++;
//                }
//            }
//        }
       temp= inter(temp1,temp2,K);
        for(int key:temp){
            if(sortedTextMap.get(key)>sortedRelationMap.get(key))
                result.put(key,sortedTextMap.get(key));//不但要得到key即所在行，而且得到预测的概率
            else
                result.put(key,sortedRelationMap.get(key));//不但要得到key即所在行，而且得到预测的概率
            // System.out.print(temp.get(i)+"  ");
        }


        //2
//        int num=0;
//        for(Map.Entry<Integer,Double> entry: sortedTextMap.entrySet()){
//            int key=entry.getKey();
//            if(sortedTextMap.get(key)>sortedRelationMap.get(key))
//            {
//                result.put(key,sortedTextMap.get(key));
//              //  System.out.println("选择的行数"+key+"其置信度为"+sortedTextMap.get(key)+"   anther"+sortedRelationMap.get(key));
//                num++;
//            }
//            if(num==K/2)
//                break;
//        }
//        num=0;
//        for(Map.Entry<Integer,Double> entry: sortedRelationMap.entrySet()){
//            int key=entry.getKey();
//            if(sortedRelationMap.get(key)>sortedTextMap.get(key))
//            {
//                result.put(key,sortedRelationMap.get(key));
//                num++;
//            }
//            if(num==K/2)
//                break;
//        }

        return result;
    }
    /**
     * 得到置信度比较高的K个样本，这K个中，K/2个来自置信度最高的，K/2来自置信度最低的
     */
    public static Map<Integer,Double> getKProp(Map<Integer,Double> sortedMap ,int K){
        if(K<0||sortedMap.size()<K){
            System.out.println("索引超出范围，请检查");
            return null;
        }
        Map<Integer,Double> result=new LinkedHashMap<>();
        List<Integer> temp=new ArrayList<>();
        for(Map.Entry<Integer,Double> entry: sortedMap.entrySet()){
            temp.add(entry.getKey());//把所有的key都加入到一个列表中
        }
        for(int i=0;i<K;i++){
            int key=temp.get(i);
            result.put(key,sortedMap.get(key));//不但要得到key即所在行，而且得到预测的概率
            // System.out.print(temp.get(i)+"  ");
        }
        return result;
    }
    /**
     * 得到置信度比较高的K个样本，这K个中，K/2个来自置信度最高的，K/2来自置信度最低的
     */
    public static Map<DataSet,Double> newgetKProp(Map<DataSet,Double> sortedMap ,int K){
        int count=0;
        Map<DataSet, Double> res=new LinkedHashMap<>();
        for(Map.Entry<DataSet,Double> entry:sortedMap.entrySet()){
//            Map<DataSet,Double> map=new HashMap<>();
//            Map<DataSet,Map<DataSet, Double>> newmap=new  HashMap<>();
//            map.put(entry.getKey(),entry.getValue());
//            newmap.put(entry.getKey(),map);
//            res.put(entry.getKey(),map);//不但要得到key即所在行，而且得到预测的概率
//             System.out.println(entry.getKey().getLabels()+"   "+entry.getValue());
            res.put(entry.getKey(),entry.getValue());
            count++;
            if(count==K)
                break;
        }
        return res;
    }
    /**
     * 对两个集合取交集
     */
    public static List<Integer> inter(List<Integer> list1,List<Integer> list2,int k){
  List<Integer> templist1;List<Integer> templist2;List<Integer> Kresult=null;
        for(int i=1;i< (list1.size()/100);i++){
            templist1=getKlist(list1,i*100);
            templist2=getKlist(list2,i*100);
            Kresult=getCommon(templist1,templist2,k);
            int size=Kresult.size();
            System.out.println("第"+i+"次迭代选择出:"+size);
            if(size==k)
                break;
        }
        return Kresult;

    }
    public static List<Integer> getKlist(List<Integer> list,int num){
        List<Integer> resultList=new ArrayList<>();
        for(int i=0;i<num;i++){
            resultList.add(list.get(i));
        }
        return resultList;
    }

    public static List<Integer> getCommon(List<Integer> list1,List<Integer> list2,int k){
        List<Integer> resultList=new ArrayList<>();
        for(int key:list1){
            if(list2.contains(key))
            {
                resultList.add(key);
            }
            if(resultList.size()==k)
                break;
        }
        return resultList;
    }
}

