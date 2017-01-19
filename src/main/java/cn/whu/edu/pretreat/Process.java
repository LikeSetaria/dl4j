package cn.whu.edu.pretreat;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Created by bczhang on 2017/1/9.
 */
public class Process {
    public static void main(String[] args){

    }

    /**
     * 处理DBLP数据集中，每行特征末尾都已经包含分类类别问题
     */
    public static void removeDBLPLabel(String DBLPDocFilePath,String savePath) throws IOException{
    List<String> lines= FileUtils.readLines(new File(DBLPDocFilePath));
        for(String line:lines){

        }
    }
}
