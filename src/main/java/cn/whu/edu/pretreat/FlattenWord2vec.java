package cn.whu.edu.pretreat;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.math.BigDecimal;
import java.util.*;

/**
 * 特征（Distributed Representations ）向量化。doc2vec
 * 1、word2vec/line等得到词向量空间
 * 2、累加每个doc中的词向量求平均，作为这个doc的特征化表示
 * Created by bczhang on 2017/1/4.
 */
public class FlattenWord2vec {

      Map<String,List<Double>> trainWordVec=new LinkedHashMap<>();//词向量，向量中每一个词已经表示为了一个向量
      Map<String,List<String>> wordsMap=new LinkedHashMap<>();//每一个doc,由那些词组成
      int K=100;//词向量的维度
       String basePath="E:\\co-training\\sample\\deeplearning4j\\textLink\\dblp_coTraining2vec\\doc_removeLabels\\";
       String trainedWordVecFile="E:\\co-training\\label_doc_vec.txt";
       String featureFile="E:\\co-training\\test_doc.txt";
       String saveLine2Vec="E:\\co-training\\test_doc_vec.txt";

       public  FlattenWord2vec(String trainWordVecFileName,String featureFileName,String saveLine2VecName){
           this.trainedWordVecFile=trainWordVecFileName;
           this.featureFile=featureFileName;
           this.saveLine2Vec=saveLine2VecName;
       }
    public static void main(String[] args) throws Exception{
        String trainedWordVecFile="E:\\co-training\\label_doc_vec.txt";
        String featureFile="E:\\co-training\\test_doc.txt";
        String  saveLine2Vec="E:\\co-training\\test_doc_vec.txt";
        FlattenWord2vec fw=new FlattenWord2vec(trainedWordVecFile,featureFile,saveLine2Vec);
        fw.flattenAvg();

    }


    /**
     * 根据已经训练好的词向量，把文档表示为特征向量。
     * 累加求平均，文档中在向量空间中不存在的使用零向量代替，得到一个文档的向量表示
     *
     */
    public  void flattenAvg() throws IOException{
        String[] word2vecLines= FileUtils.readFileToString(new File(trainedWordVecFile)).toUpperCase().split("\n");
        String[] featureLines= FileUtils.readFileToString(new File(featureFile)).split("\n");
        //System.out.println(word2vecLines.length);
        for(int i=0;i<word2vecLines.length;i++){
            List<Double> aList=new ArrayList<>();
            String [] arr=word2vecLines[i].split("\\s+");
           for(int j=1;j<arr.length;j++){
               aList.add(Double.valueOf(arr[j]));
           }
          // System.out.println("初始化的词"+arr[0]+" "+aList);
            trainWordVec.put(arr[0],aList);
        }
        for(String word:featureLines){
            String[] ar=word.split("\\s+");
            List<String> aList=new ArrayList<>();
            for(int j=1;j<ar.length;j++){
               // if(trainWordVec.containsKey(ar[j]))
                aList.add(ar[j]);
            }
            wordsMap.put(ar[0],aList);
           //System.out.println(ar[0]+"  "+aList);
        }
        averageVec();
    }
    /**
     * 文档特征向量化，求平均
     */
    private  void averageVec() throws IOException{
        StringBuilder strb=new StringBuilder();
        for(Map.Entry<String,List<String>> entry:wordsMap.entrySet()){
        List<List<Double>> vec=new ArrayList<>();
            String key=entry.getKey();
            List<String> value=entry.getValue();
            for(int i=0;i<value.size();i++){
                if(trainWordVec.containsKey(value.get(i))) {
                    vec.add(trainWordVec.get(value.get(i)));
                    //System.out.println(value.get(i) + "*" + trainWordVec.get(value.get(i)));
                }
//                else { vec.add(trainWordVec.get("ZREOVEC"));
//                    System.out.println(value.get(i) + "目标词，向量空间中不存在，拼接零向量");
//                }
            }
            //System.out.println(vec);
            if(value.size()==0)
                System.out.print(key+"  "+ value);
            else{
            strb.append(key+","+matrixSum(vec,value.size()));
            strb.append("\n");
            }
           //System.out.println(key+"  "+matrixSum(vec));
        }
         FileUtils.write(new File(saveLine2Vec),strb.toString().replace("[","").replace("]","").replace(" ",""));
        //System.out.println(strb.toString().replace("[","").replace("]",""));
    }
    /**
     * 计算n行,K维矩阵,行累加平均值
     * 1，2，3
     * 1，1，0
     * =（2，3，3）/2
     * @param list 多少行
     */
   private   List<Double> matrixSum(List<List<Double>> list,int featuresCount){
       double[] accArr=new double[K];//默认初始化为0.0的，如果是Double 是Object,数组默认初始化为null
      // System.out.println(" 组合词数 "+list.size());
       List<Double> result=new ArrayList<>();
       for(List<Double> dd:list){
           for(int j=0;j<accArr.length;j++){
               accArr[j]+=dd.get(j);
              // System.out.print(accArr[j]+" ");
           }
       }
       for(double d:accArr){
           BigDecimal   b   =   new   BigDecimal(d/featuresCount);
           double   f1   =   b.setScale(6,   BigDecimal.ROUND_HALF_UP).doubleValue();
           result.add(f1);
       }
       //System.out.println(result);
    return result;
   }
}
