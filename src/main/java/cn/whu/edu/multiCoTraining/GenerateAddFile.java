package cn.whu.edu.multiCoTraining;

import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * 根据已经选定K个标签，生成model 更新的添加文件
 *
 * Created by 宝超 on 2017/2/19.
 */
public class GenerateAddFile {
    String LabelFilePath="";
    //由于写在静态代码块中，导致构造函数之前都要初始化路径参数
     String UnLabelFilePath="";
     String addFilePath="";
    private   Map<Integer,String>UnlabelFileMap=new HashMap<>();
    //一次过程，要频繁的更新，由于文件不大，这里处理它常驻内存加快速度。静态块，不合适
    private void init(){
        try {
            System.out.println("根据K个置信度高的文件，生成对应的更新w2v的添加文件");
            String[] UnlabeledFilelines = FileUtils.readFileToString(new File(UnLabelFilePath)).split("\n");
            int lineNum=1;
            for(String line:UnlabeledFilelines){
                UnlabelFileMap.put(lineNum++,line);
            }
        }catch(IOException e){
            throw new RuntimeException(e);
        }
    }
    public GenerateAddFile(String LabelFilePath, String UnLabelFilePath,String addFilePath){
        this.LabelFilePath=LabelFilePath;
        this.UnLabelFilePath=UnLabelFilePath;
        this.addFilePath=addFilePath;
    }

    /**
     * 根据选择置信度高的K行，
     * @param selectedAdd map中key对应的是文件中的行号，文件行号从1开始
     * @return
     * @throws IOException
     */
    public String updateFile(Map<Integer,DataSet> selectedAdd) throws IOException{
        init();

        //得到增加到Label File 的doc行号
        Set<Integer> keys=selectedAdd.keySet();
        StringBuilder serb=new StringBuilder();
        for(Integer i:keys){
//            System.out.println(i+"    ***更新添加的问文件*****************    "+UnlabelFileMap.get(i));
            serb.append(UnlabelFileMap.get(i));
            serb.append("\n");
        }
        FileUtils.write(new File(addFilePath),serb);
        return serb.toString();

    }
    public static void main(String[] args) throws IOException{
          String LabelFilePath="E:\\co-training\\label_doc.txt";
          String UnLabelFilePath="E:\\co-training\\unlabel_doc.txt";
          String addFilePath="E:\\co-training\\label_doc_add.txt";
        GenerateAddFile ulf=new GenerateAddFile(LabelFilePath,UnLabelFilePath,addFilePath);
        Map<Integer,DataSet> testMap=new HashMap<>();
        testMap.put(100,null);
        testMap.put(120,null);
        testMap.put(15,null);
        testMap.put(1900,null);
        testMap.put(1000,null);
        ulf.updateFile(testMap);

    }
}
