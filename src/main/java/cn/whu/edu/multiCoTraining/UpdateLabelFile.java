package cn.whu.edu.multiCoTraining;

import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * 根据已经选定的序列，更新Lable文件
 *
 * Created by 宝超 on 2017/2/19.
 */
public class UpdateLabelFile {
    static  String LabelFilePath="";
    //由于写在静态代码块中，导致构造函数之前都要初始化路径参数
    static String UnLabelFilePath="E:\\co-training\\unlabel_doc.txt";
    private  static Map<Integer,String>UnlabelFileMap=new HashMap<>();
    //一次过程，要频繁的更新，由于文件不大，这里处理它常驻内存加快速度
    static{
        try {
            System.out.println("初始化更新文件程序");
            String[] UnlabelFilelines = FileUtils.readFileToString(new File(UnLabelFilePath)).split("\n");
            int lineNum=1;
            for(String line:UnlabelFilelines){
                UnlabelFileMap.put(lineNum++,line);
            }
        }catch(IOException e){
            throw new RuntimeException(e);
        }
    }
    public UpdateLabelFile(String LabelFilePath,String UnLabelFilePath){
        this.LabelFilePath=LabelFilePath;
        this.UnLabelFilePath=UnLabelFilePath;
    }

    /**
     * 根据选择置信度高的K行，
     * @param selectedAdd
     * @return
     * @throws IOException
     */
    public boolean updateFile(Map<Integer,DataSet> selectedAdd) throws IOException{
        boolean result=false;
        //得到增加到Label File 的doc行号
        Set<Integer> keys=selectedAdd.keySet();
        StringBuilder strb=new StringBuilder();
        for(Integer i:keys){
            strb.append(UnlabelFileMap.get(i));
            strb.append("\n");
        }
        FileUtils.write(new File(LabelFilePath),strb,true);
        return result;
    }
    public static void main(String[] args) throws IOException{
          String LabelFilePath="E:\\co-training\\label_doc.txt";
          String UnLabelFilePath="E:\\co-training\\unlabel_doc.txt";
        UpdateLabelFile ulf=new UpdateLabelFile(LabelFilePath,UnLabelFilePath);
        Map<Integer,DataSet> testmap=new HashMap<>();
        testmap.put(100,null);
        testmap.put(120,null);
        testmap.put(15,null);
        testmap.put(1900,null);
        testmap.put(1000,null);
        ulf.updateFile(testmap);

    }
}
