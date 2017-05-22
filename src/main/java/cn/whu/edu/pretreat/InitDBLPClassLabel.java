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
    private Map<Integer,Integer> Allselected=new LinkedHashMap<>();
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
    public void replaceLabel(Map<Integer,Integer> selected,String unlabeledSourceFilePath) throws IOException{
        String [] idLabels= FileUtils.readFileToString(new File(labelsFilePath)).trim().split("\n");
        String [] targetVec= FileUtils.readFileToString(new File(targetFilePath)).trim().split("\n");
       String [] unlabeled= FileUtils.readFileToString(new File(unlabeledSourceFilePath)).trim().split("\n");
        Map<Integer,Integer> lineNum_id=new LinkedHashMap<>();
        int count=1;//行号作为主键，依然是从序号1开始
        //行号和文档ID做一个映射
        for(String s:unlabeled){
            String[] ar=s.split("\\s+");
            lineNum_id.put(count++,Integer.valueOf(ar[0].trim()));
        }
        //每次添加的保存到一个map中
        System.out.println("本次迭代添加前的已选择大小为"+this.Allselected.size());
        //这这次迭代需要添加的，添加到里面总的已经选择去
        this.Allselected.putAll(selected);
        System.out.println("本次迭代添加后后后后的已选择大小为"+this.Allselected.size());
        Map<Integer,Integer> key=new LinkedHashMap<>();
        Set<String> idset=new LinkedHashSet<>();
        for(String line:idLabels){
            String[] arr=line.split("\\s");
            key.put(Integer.valueOf(arr[1].trim()), Integer.valueOf(arr[0].trim()));
        }
        for(Map.Entry<Integer,Integer> entry:Allselected.entrySet()){
            int rowNo2ID=lineNum_id.get(entry.getKey());
            int predictedLabel=entry.getValue();
          //  System.out.println(rowNo2ID+"更新前"+key.get(rowNo2ID));
            //更新id_label映射，因为后面是根据这个映射进行Label.txt的ID

            key.put(rowNo2ID,predictedLabel);
            //System.out.println(rowNo2ID+"更新后"+key.get(rowNo2ID));

        }

        StringBuilder str=new StringBuilder();
        for(String line:targetVec){
            String []arr=line.trim().split(",");
            if(key.get(Integer.valueOf(arr[0].trim()))!=null)
            str.append(line.replace(arr[0],key.get(Integer.valueOf(arr[0].trim())).toString()));
            else System.out.println("cuocuocuoccuoucoouccuocuoccuucocuo");
            str.append("\n");
        }
        FileUtils.write(new File(saveFilePath),str);
    }


    public static void main(String[] args)throws Exception{
//          String labelsFilePath="E:\\co-training\\selected_id_classLabel.txt";
//          String targetFilePath="E:\\co-training\\test_doc_vec.txt";
//          String saveFilePath="E:\\co-training\\test_doc_vec_id.txt";
          String labelsFilePath="E:\\co-training\\selected_id_classLabel.txt";
          String targetFilePath="E:\\co-training\\sample\\deeplearning4j\\textLink\\dblp_coTraining2vec\\test_doc2vec.txt";
          String saveFilePath= "E:\\co-training\\sample\\deeplearning4j\\textLink\\dblp_coTraining2vec\\test_doc2vec_id.txt";
          InitDBLPClassLabel init=new InitDBLPClassLabel(labelsFilePath,targetFilePath,saveFilePath);

        init.replaceLabel();

    }

}
