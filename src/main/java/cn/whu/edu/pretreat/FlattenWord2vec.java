package cn.whu.edu.pretreat;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * 平均化特征向量
 * Created by bczhang on 2017/1/4.
 */
public class FlattenWord2vec {
    static Map<String,List<Double>> wordVec=new LinkedHashMap<>();
    static Map<String,List<String>> wordsMap=new LinkedHashMap<>();
    static  final String basePath="E:\\co-training\\sample\\deeplearning4j\\textLink\\dblp_coTraining2vec\\";
    static  final String wordVecFileName="label_link2vec.txt";
    static  final String wordsFileName="label_link.txt";
    public static void main(String[] args) throws  Exception{
       // initWord2Map();
        //averageVec();
        List<List<Double>> listlist=new LinkedList<>();
        List<Double> list1=new ArrayList<>();
        List<Double> list2=new ArrayList<>();
        List<Double> list3=new ArrayList<>();
        for(int i=0;i<10;i++){
            list1.add(0.1*(int)(1+Math.random()*(10-1+1)));
            list2.add(0.2*(int)(1+Math.random()*(10-1+1)));
            list3.add(0.3*(int)(1+Math.random()*(10-1+1)));
        }
        System.out.println(list1);
        System.out.println(list2);
        System.out.println(list3);
        listlist.add(list1);
        listlist.add(list2);
        listlist.add(list3);
        acc(listlist);
    }

    /**
     * 初始化 词向量
     */
    public static  void initWord2Map() throws IOException{
        String[] lines= FileUtils.readFileToString(new File(basePath+wordVecFileName)).split("\n");
        String[] words= FileUtils.readFileToString(new File(basePath+wordsFileName)).split("\n");
        for(int i=1;i<lines.length;i++){
            List<Double> aList=new ArrayList<>();
            String [] arr=lines[i].split("\\s");
           for(int j=1;j<arr.length;j++){
               aList.add(Double.valueOf(arr[j]));
           }
            wordVec.put(arr[0],aList);

        }


        for(String word:words){
            String[] ar=word.split("\\s");
            List<String> aList=new ArrayList<>();
            for(int j=1;j<ar.length;j++){
                aList.add(ar[j]);
            }
            wordsMap.put(ar[0],aList);
           // System.out.println(ar[0]+"  "+aList);
        }

    }
    /**
     *
     */
    public static void averageVec() throws IOException{
        for(Map.Entry<String,List<String>> entry:wordsMap.entrySet()){
            String key=entry.getKey();
            List<String> value=entry.getValue();
            System.out.println(key);
            Double arr[]=new Double[100];
            for(int i=0;i<value.size();i++){
                System.out.println(wordVec.get(i));

            }
        }
    }
   public static  void acc(List<List<Double>> list){
       Double[] accArr=new Double[10];
       for(List<Double> dd:list){

           for(int i=0;i<dd.size();i++){
               System.out.println(dd.get(i));
               accArr[i]=dd.get(i)+accArr[i];

           }
       }
//       for(Double d:accArr){
//           System.out.print(d/list.size());
//           System.out.print(" ");
//       }

   }

}
