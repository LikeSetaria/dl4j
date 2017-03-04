package cn.whu.edu.pretreat;

import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * DBLP数据集，doc Id映射到分类标签
 * Created by 宝超 on 2017/2/21.
 */
public class InitDBLPClassLabel {
    private String labelsFilePath="";
    private String targetFilePath="";
    private String saveFilePath="";
    public InitDBLPClassLabel(String labelsFilePath,String targetFilePath,String saveFilePath){
        this.labelsFilePath=labelsFilePath;
        this.targetFilePath=targetFilePath;
        this.saveFilePath=saveFilePath;

    }

    /**
     *根据类标签对应文件更新向量类别编号
     */
    public void replaceLabel() throws IOException{
        String [] idLabels= FileUtils.readFileToString(new File(labelsFilePath)).trim().split("\n");
        String [] targetVec= FileUtils.readFileToString(new File(targetFilePath)).trim().split("\n");



        Map<String,String> key=new HashMap<>();
        for(String line:idLabels){
            String[] arr=line.split("\\s");
            key.put(arr[1].trim(), arr[0].trim());
        }
        StringBuilder str=new StringBuilder();
        for(String line:targetVec){
            String []arr=line.trim().split(",");
            str.append(line.replace(arr[0],key.get(arr[0].trim())));
            str.append("\n");
        }
        FileUtils.write(new File(saveFilePath),str);
    }

    /**
     * 增量更新时，得更新分类器预测的类标签，而不能用本身的（因为假设他们是Unlabeled）
     * @param selected 这里的key是无标签的行号，不是doc id所以还需要进行转化
     * @throws IOException
     */
    public void replaceLabel(Map<Integer,DataSet> selected,String unlabeledSourceFilePath) throws IOException{
        String [] idLabels= FileUtils.readFileToString(new File(labelsFilePath)).trim().split("\n");
        String [] targetVec= FileUtils.readFileToString(new File(targetFilePath)).trim().split("\n");
        String [] unlabeled= FileUtils.readFileToString(new File(unlabeledSourceFilePath)).trim().split("\n");
        Map<Integer,String> lineNum_id=new HashMap<>();
        int count=1;
        for(String s:unlabeled){
            String[] ar=s.split("\\s+");
            lineNum_id.put(count++,ar[0].trim());
        }



        Map<String,String> key=new LinkedHashMap<>();
        Set<String> idset=new LinkedHashSet<>();
        for(String line:idLabels){
            String[] arr=line.split("\\s");
            key.put(arr[1].trim(), arr[0].trim());
        }
        StringBuilder str=new StringBuilder();
        for(String line:targetVec){
            String []arr=line.trim().split(",");
            str.append(line.replace(arr[0],key.get(arr[0].trim())));
            str.append("\n");
        }
        FileUtils.write(new File(saveFilePath),str);
    }


    public static void main(String[] args)throws Exception{
          String labelsFilePath="E:\\co-training\\selected_id_classLabel.txt";
          String targetFilePath="E:\\co-training\\test_doc_vec.txt";
          String saveFilePath="E:\\co-training\\test_doc_vec_id.txt";
          InitDBLPClassLabel init=new InitDBLPClassLabel(labelsFilePath,targetFilePath,saveFilePath);
          init.replaceLabel();

    }

}
