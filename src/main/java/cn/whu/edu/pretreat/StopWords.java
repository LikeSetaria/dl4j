package cn.whu.edu.pretreat;

import org.apache.commons.io.FileUtils;
import org.apache.commons.logging.Log;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by bczhang on 2016/12/19.
 */
public class StopWords {
    private static Logger log = LoggerFactory.getLogger(StopWords.class);
    public static void main(String[] args) throws Exception{
        processStopWrods();

    }

    /**
     * 停用词处理，去除文档中的停用词
     */
    public static  void processStopWrods() throws IOException{
        String text= FileUtils.readFileToString(new File("E:\\co-training\\sample\\deeplearning4j\\textLink\\cora\\stopWordsCoraText.txt"));
        String []lines=text.split("\n");
        String stopToWordsText=FileUtils.readFileToString(new File("D:\\bczhang\\workspace\\ideaWorkplace\\dl4j-examples\\dl4j-examples\\src\\main\\resources\\stopWords.txt"));
        String wordsLines[]=stopToWordsText.split(",");
        Set<String> wordsSet=new HashSet<>();
        StringBuilder result=new StringBuilder();
        log.info("初始化停用词列表");
        for(String s:wordsLines){
            wordsSet.add(s.trim());
        }
        for(String line:lines){
            StringBuilder strb=new StringBuilder();
            String [] arr=line.replaceAll( "[\\pP+~$`^=|<>～｀＄＾＋＝｜＜＞￥×]" , "").split("\\s");
            System.out.println(line);
            for(String word:arr){
                if(!wordsSet.contains(word)){
                    strb.append(word);
                    strb.append(" ");
                }
            }
            result.append(strb);
            result.append("\n");
        }
      //  FileUtils.write(new File(("E:\\co-training\\sample\\deeplearning4j\\textLink\\cora\\stopWordsCoraText.txt")),result);

    }
}
