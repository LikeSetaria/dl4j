package cn.whu.edu.pretreat;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.List;

/**
 * Created by bczhang on 2017/1/6.
 */
public class Scale {
    public static void main(String[] args) throws Exception{
        String basePath="E:/co-training/sample/deeplearning4j/textLink/dblp/";
 String fileName="test_data1.csv";
        String saveName="test_data1_scale.csv";
        //scaleVec(basePath+fileName,basePath+saveName);
        processInterval("E:\\co-training\\source\\textLink\\DBLP\\dblplink.collection","E:\\co-training\\source\\textLink\\DBLP\\dblplink.txt");
    }
    /**
     *对特征向量，保留固定小数处理
     */
    public static void scaleVec(String path,String save) throws IOException{
        List<String> lines= FileUtils.readLines(new File(path));
        DecimalFormat df=new   java.text.DecimalFormat("#.######");
        StringBuilder str=new StringBuilder();
        for(String line:lines){
        String[] arr=line.split(",");
            str.append(arr[0]);
            str.append(",");
            for(int i=1;i<arr.length;i++){
                str.append(df.format(Double.valueOf(arr[i])));
                str.append(",");

            }
            str.append("\n");

        }
        System.out.println(str);
        FileUtils.write(new File(save),str);
    }

    /**
     * 预处理，去除间隔字段
     */
    public static void processInterval(String path,String save) throws IOException{
        List<String> lines= FileUtils.readLines(new File(path));
        StringBuilder strb=new StringBuilder();
        for(String line:lines){
            String[] arr=line.split("\t");
            strb.append(arr[0]);
            strb.append("\t");
            strb.append(arr[1]);

            for(int i=2;i<arr.length;i++){
           if(!arr[i].equals("1")) {
               strb.append("\t");
               strb.append(arr[i]);

           }
            }
            strb.append("\n");
        }
        FileUtils.write(new File(save),strb);
        System.out.println(strb);
    }
}
