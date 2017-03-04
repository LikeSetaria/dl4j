package cn.whu.edu.multiCoTraining;

import cn.whu.edu.pretreat.FlattenWord2vec;
import org.apache.commons.io.FileUtils;

import java.io.File;

/**
 * Created by 宝超 on 2017/2/16.
 * 得到文档向量表示
 */
public class GetUpTrainFeatures {
    private  String initFilePath="E:\\co-training\\trainW2v\\label_doc.txt";
    private String addFilePath="E:\\co-training\\trainW2v\\label_doc_add.txt";
    private String saveModelPath="E:\\co-training\\trainW2v\\label_doc.w2vModel";
    private String saveW2vPath="E:\\co-training\\trainW2v\\label_doc_vec.txt";

    private String featureFile="E:\\co-training\\test_doc.txt";
    private String saveLine2Vec="E:\\co-training\\test_doc_vec.txt";
    public GetUpTrainFeatures(String initFilePath,String addFilePath,
                              String saveModelPath,String saveW2vPath,
                              String featureFile,String saveLine2Vec
    ){
        this.initFilePath=initFilePath;
        this.addFilePath=addFilePath;
        this.saveModelPath=saveModelPath;
        this.saveW2vPath=saveW2vPath;
        this.featureFile=featureFile;
        this.saveLine2Vec=saveLine2Vec;
    }
    public void get() throws  Exception{
        System.out.println("得到文档的向量表示");
        FlattenWord2vec flatW2v=new FlattenWord2vec(saveW2vPath,featureFile,saveLine2Vec);
        IncrementalTrainWord2vec increTrain=new IncrementalTrainWord2vec(initFilePath,addFilePath,saveW2vPath,saveModelPath);

        //increTrain.upTrain();
        increTrain.trainW2V();
        flatW2v.flattenAvg();

    }
    public static void main(String[] args)throws Exception{
        String initFilePath="E:\\co-training\\label_doc.txt";
        String addFilePath="E:\\co-training\\label_doc_add.txt";
        String saveModelPath="E:\\co-training\\label_doc.w2vModel";
        String saveW2vPath="E:\\co-training\\label_doc_vec.txt";

        String featureFile="E:\\co-training\\test_doc.txt";
        String saveLine2Vec="E:\\co-training\\test_doc_vec.txt";
        GetUpTrainFeatures up=new GetUpTrainFeatures(  initFilePath,  addFilePath,
                  saveModelPath, saveW2vPath,
                  featureFile, saveLine2Vec);
        up.get();
    }
}

