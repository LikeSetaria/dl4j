package cn.whu.edu.pretreat;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * 特征（Distributed Representations ）向量化。doc2vec
 * 1、word2vec/line等得到词向量空间
 * 2、累加每个doc中的词向量求平均，作为这个doc的特征化表示
 * Created by bczhang on 2017/1/4.
 */
public class FlattenWord2vec {

    static Map<String,List<Double>> wordVec=new LinkedHashMap<>();//词向量，向量中每一个词已经表示为了一个向量
    static Map<String,List<String>> wordsMap=new LinkedHashMap<>();//doc,组成的词有那些
    static final int K=100;//词向量的维度
    static  final String basePath="E:\\co-training\\sample\\deeplearning4j\\textLink\\dblp_coTraining2vec\\";
    static  final String wordVecFileName="test_docWord2vec.txt";
    static  final String featureFileName="test_doc.txt";
    static final String  saveLine2VecName="test_doc2vec.txt";
    public static void main(String[] args) throws  Exception{
       initWord2Map();
        averageVec();
    }

    /**
     * 初始化 词向量
     */
    public static  void initWord2Map() throws IOException{
        String[] lines= FileUtils.readFileToString(new File(basePath+wordVecFileName)).split("\n");
        String[] words= FileUtils.readFileToString(new File(basePath+featureFileName)).split("\n");
        for(int i=1;i<lines.length;i++){
            List<Double> aList=new ArrayList<>();
            String [] arr=lines[i].split("\\s+");
           for(int j=1;j<arr.length;j++){
               aList.add(Double.valueOf(arr[j]));
           }
            wordVec.put(arr[0],aList);

        }


        for(String word:words){
            String[] ar=word.split("\\s+");
            List<String> aList=new ArrayList<>();
            for(int j=1;j<ar.length;j++){
                aList.add(ar[j]);
            }
            wordsMap.put(ar[0],aList);
           // System.out.println(ar[0]+"  "+aList);
        }

    }
    /**
     *求平均
     */
    public static void averageVec() throws IOException{
        StringBuilder strb=new StringBuilder();
        for(Map.Entry<String,List<String>> entry:wordsMap.entrySet()){
        List<List<Double>> vec=new ArrayList<>();
            String key=entry.getKey();
            List<String> value=entry.getValue();
            for(int i=0;i<value.size();i++){
                vec.add(wordVec.get(value.get(i)));
                //System.out.println(value.get(i)+"*"+wordVec.get(value.get(i)));
            }
            strb.append(key+","+matrixSum(vec));
            strb.append("\n");
           // System.out.println(key+"  "+matrixSum(vec));
        }
         FileUtils.write(new File(basePath+saveLine2VecName),strb.toString().replace("[","").replace("]",""));
        //System.out.println(strb.toString().replace("[","").replace("]",""));
    }

    /**
     * 计算n行，K维矩阵，行累加平均值
     * 1，2，3
     * 1，1，0
     * =（2，3，3）/2
     * @param list 多少行
     */
   public static  List<Double> matrixSum(List<List<Double>> list){
       double[] accArr=new double[K];//默认初始化为0.0的，如果是Double 是Object,数组默认初始化为null
       List<Double> result=new ArrayList<>();
       for(List<Double> dd:list){
           for(int j=0;j<accArr.length;j++){
               accArr[j]+=dd.get(j);
           }
       }
       for(double d:accArr){
           result.add(d/list.size());
       }
    return result;
   }

}
