package cn.whu.edu.pretreat;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created by bczhang on 2017/1/9.
 */
public class Process {
    public static void main(String[] args)throws  IOException{
        File file=new File("D:\\DOWNLOAD\\BaiduYunDownload\\Classification1\\Classification1");
        String file2[]=FileUtils.readFileToString(new File("D:\\DOWNLOAD\\BaiduYunDownload\\Classification1\\doc.txt")).split("\n");
        String[] str=file.list();
        Set<String> set1=new HashSet<>();
        Set<String> set2=new HashSet<>();
        for(String s:str){
        set1.add(s.trim());

        }
        for(String s:file2){
            set2.add(s.trim());
            //System.out.println(s);
        }

        for(String s:file2){
            if(!set1.contains(s))
                System.out.println(s);
        }
    }

    /**
     * 处理DBLP数据集中，每行特征末尾都已经包含分类类别问题
     */

}
