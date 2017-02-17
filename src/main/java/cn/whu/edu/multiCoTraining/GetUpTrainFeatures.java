package cn.whu.edu.multiCoTraining;

import cn.whu.edu.pretreat.FlattenWord2vec;

/**
 * Created by 宝超 on 2017/2/16.
 * 得到文档向量表示
 */
public class GetUpTrainFeatures {
    String initFilePath="E:\\co-training\\trainW2v\\label_doc.txt";
    String addFilePath="E:\\co-training\\trainW2v\\label_doc_add.txt";
    String saveModelPath="E:\\co-training\\trainW2v\\label_doc.w2vModel";
    String saveW2vPath="E:\\co-training\\trainW2v\\label_doc_vec.txt";

    String featureFile="E:\\co-training\\test_doc.txt";
    String saveLine2Vec="E:\\co-training\\test_doc_vec.txt";
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
        FlattenWord2vec flatW2v=new FlattenWord2vec(saveW2vPath,featureFile,saveLine2Vec);
        IncrementalTrainWord2vec increTrain=new IncrementalTrainWord2vec(initFilePath,addFilePath,saveW2vPath,saveModelPath);

        increTrain.upTrain();
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

