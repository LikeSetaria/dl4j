package cn.whu.edu.cotraining;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.util.*;

/**
 * Created by bczhang on 2016/12/11.
 */
public class GetLinkVec {
    public static void main(String[] args) throws Exception{
        String path="E:\\co-training\\sample\\deeplearning4j\\textLink\\";
        String [] id= FileUtils.readFileToString(new File(path+"id.txt")).trim().split("\n");
        String [] idLabels= FileUtils.readFileToString(new File(path+"selected_id_classLabel.txt")).trim().split("\n");
        String [] linkVec= FileUtils.readFileToString(new File(path+"selected_link_vec_sort.txt")).trim().split("\n");
        String  []doc= FileUtils.readFileToString(new File(path+"selected_doc_vec_sort.txt")).trim().split("\n");
        StringBuilder strb=new StringBuilder();
        Map<String,String> linkMap=new HashMap<>();
        Map<String,String> key=new HashMap<>();
        Set<String> idset=new LinkedHashSet<>();
        for(String s:id){
            idset.add(s.trim());
        }
        for(String line:idLabels){
           String[] arr=line.split("\\s");
            key.put(arr[1].trim(), arr[0].trim());
        }
        StringBuilder str=new StringBuilder();
        for(String line:linkVec){
            String []arr=line.trim().split("\\s+");
           // System.out.println(arr[0]+" "+key.get(arr[0].trim()));
           // linkMap.put(arr[0],line.trim());
            str.append(line.replace(arr[0],key.get(arr[0].trim())));
            str.append("\n");
        }

//        for(String is:idset){
//            if(key.containsKey(is))
//            if(key.get(is).equals("DB")||key.get(is).equals("ML")){
//                System.out.println(key.get(is)+"  "+is);
//            str.append(linkMap.get(is));
//            str.append("\n");
//            }
//        }
       FileUtils.write(new File(path+"selected_link_vec_id.txt"),str);
//        java.text.DecimalFormat   df=new   java.text.DecimalFormat("#.######");
//        for(String line:doc){
//            String [] arr=line.split("\\s");
//            StringBuilder st=new StringBuilder();
//            st.append(arr[0]);
//            st.append(",");
//            for(int i=1;i<101;i++){
//                String temp=arr[i].replace(i+":","");
//               st.append(df.format(Double.valueOf(temp)));
//                st.append(",");
//            }
//            System.out.println(st);
//           // st.append(key.get(arr[0]));
//            strb.append(st);
//            strb.append("\n");
//        }
//        FileUtils.write(new File(path+"selected_doc_vec_classid.txt"),strb);
    }

}
