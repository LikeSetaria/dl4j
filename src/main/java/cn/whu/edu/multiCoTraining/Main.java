package cn.whu.edu.multiCoTraining;

import cn.whu.edu.pretreat.FlattenWord2vec;
import cn.whu.edu.pretreat.InitDBLPClassLabel;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.examples.bczhang.LineChartsTest;
import org.deeplearning4j.examples.bczhang.NNconf;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.jfree.ui.RefineryUtilities;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Created by bczhang on 2016/12/20.
 */
public class Main {
    public static  void  main(String[] agrs) throws Exception{
        reset();
        classicCoT();
        //mergeCoT();
        //getUnlabeledTest_id();
    }
//    public static void mergeCoT()throws  Exception{
//        Map<String,List<Double>> resultMap=new HashMap<>();
//        //  训练model
//        NNModel graphTrainModel=new NNModel("L_data2","Test_data2");
//        graphTrainModel.init();
//        NNModel TextTrainModel=new NNModel("L_data1","Test_data1");
//        TextTrainModel.init();
//        // K置信度选择器
//        TopConfidence graphChoosePropIdex=new TopConfidence("U_data2","U_data1");
//        //TopConfidence textChoosePropIdex=new TopConfidence("U_data1","U_data2");
//        graphChoosePropIdex.init();
//        // textChoosePropIdex.init();
//        //  设置不同的配置文件
//        NNconf graphnnconf=new NNconf(12);
//
//        NNconf textnnconf=new NNconf(16);
//
//        MultiLayerConfiguration graphconf=graphnnconf.getConf();
//        MultiLayerConfiguration textconf=textnnconf.getConf();
//        List<DataSet> L_list1=null;
//        List<DataSet> L_list2=null;
//        MultiLayerNetwork model_graph=new MultiLayerNetwork(graphconf);
//        model_graph=graphTrainModel.GetNNModel("L_data2",L_list2);
//        MultiLayerNetwork model_Text=new MultiLayerNetwork(textconf);
//        model_Text=TextTrainModel.GetNNModel("L_data1",L_list1);
//        Map<String,List<DataSet>> KPropMaps=null;
//        for(int i=1;i<=2;i++) {
//            KPropMaps=graphChoosePropIdex.getKPropIndex(model_graph,model_Text,10);
//            L_list1 = KPropMaps.get("text");
//            L_list2 =KPropMaps.get("relation");
//            model_graph = graphTrainModel.GetNNModel("L_data2", L_list2);
//            model_Text=TextTrainModel.GetNNModel("L_data1",L_list1);
//        }
//        // training();
//     //   resultMap.put("关系ACC",graphTrainModel.acc);
//        resultMap.put("关系F1",graphTrainModel.f1);
//      //  resultMap.put("文本ACC",TextTrainModel.acc);
//        resultMap.put("文本F1",TextTrainModel.f1);
//        //System.out.println("关系ACC："+graphTrainModel.acc);
//        System.out.println("关系F1："+graphTrainModel.f1);
//        //System.out.println("文本ACC："+TextTrainModel.acc);
//        System.out.println("文本F1："+TextTrainModel.f1);
//
//        LineChartsTest fjc = new LineChartsTest("折线图",resultMap);
//        fjc.pack();
//        RefineryUtilities.centerFrameOnScreen(fjc);
//        fjc.setVisible(true);
//    }
    public static void classicCoT() throws  Exception{
        Map<String,List<Double>> resultMap=new HashMap<>();
        //  训练model
        NNModel graphTrainModel=new NNModel("L_data2","Test_data2");
        graphTrainModel.init();
        NNModel TextTrainModel=new NNModel("L_data1","Test_data1");
        TextTrainModel.init();
        // K置信度选择器
        TopConfidence graphChoosePropIdex=new TopConfidence("U_data2","U_data1");
        TopConfidence textChoosePropIdex=new TopConfidence("U_data1","U_data2");
        graphChoosePropIdex.init();
        textChoosePropIdex.init();
        //  设置不同的配置文件
        List<DataSet> L_list1=null;
        List<DataSet> L_list2=null;
        Map<Integer,DataSet> map_graph;
        Map<Integer,DataSet> map_text;
//        List<Integer> L_list1=null;
//        List<Integer> L_list2=null;
//        Map<Integer,Integer> map_graph;
//        Map<Integer,Integer> map_text;
        //a、标签原始数据路径、无标签原始数据路径
        String LabelFilePath_graph="E:\\co-training\\trainW2v\\link\\label_link.txt";
        String UnlabeledFilePath_graph="E:\\co-training\\trainW2v\\link\\unlabel_link.txt";
        String addFilePath_graph="E:\\co-training\\trainW2v\\link\\label_link_add.txt";

        String LabelFilePath_text="E:\\co-training\\trainW2v\\doc\\label_doc.txt";
        String UnlabeledFilePath_text="E:\\co-training\\trainW2v\\doc\\unlabel_doc.txt";
        String addFilePath_text="E:\\co-training\\trainW2v\\doc\\label_doc_add.txt";
        GenerateAddFile generateAddFile_graph =new GenerateAddFile(LabelFilePath_graph,UnlabeledFilePath_graph,addFilePath_graph);
        GenerateAddFile generateAddFile_text =new GenerateAddFile(LabelFilePath_text,UnlabeledFilePath_text,addFilePath_text);
        //b、增量更新word2vec文件路径
        String initFilePath_graph=LabelFilePath_graph;
        String saveModelPath_graph="E:\\co-training\\trainW2v\\link\\label_link.w2vModel";
        String saveW2vPath_graph="E:\\co-training\\trainW2v\\link\\label_link.w2v";

        String initFilePath_text=LabelFilePath_text;
        String saveModelPath_text="E:\\co-training\\trainW2v\\doc\\label_doc.w2vModel";
        String saveW2vPath_text="E:\\co-training\\trainW2v\\doc\\label_doc.w2v";

        //IncrementalTrainWord2vec upTrainW2v_graph=new IncrementalTrainWord2vec(initFilePath_graph,addFilePath_graph,saveW2vPath_graph,saveModelPath_graph);
       // IncrementalTrainWord2vec upTrainW2v_text=new IncrementalTrainWord2vec(initFilePath_text,addFilePath_text,saveW2vPath_text,saveModelPath_text);
        //c、由最新的词向量，得到新的特征向量表示

        String featureFile_graph="E:\\co-training\\trainW2v\\link\\label_link.temp";
        String saveLine2Vec_graph="E:\\co-training\\trainW2v\\link\\label_link_vec.txt";
        String featureFile_text="E:\\co-training\\trainW2v\\doc\\label_doc.temp";
        String saveLine2Vec_text="E:\\co-training\\trainW2v\\doc\\label_doc_vec.txt";
        //无标签，和测试数据也得根据，label得到的词向量进行重新向量化

        GetUpTrainFeatures getUpTrainFeatures_graph=new GetUpTrainFeatures(initFilePath_graph,  addFilePath_graph,
                saveModelPath_graph, saveW2vPath_graph,
                featureFile_graph, saveLine2Vec_graph);
        GetUpTrainFeatures getUpTrainFeatures_text=new GetUpTrainFeatures(initFilePath_text,  addFilePath_text,
                saveModelPath_text, saveW2vPath_text,
                featureFile_text, saveLine2Vec_text);
        //d、更新类标签
          String labelsFilePath="E:\\co-training\\selected_id_classLabel.txt";
          String targetFilePath_graph=saveLine2Vec_graph;
          String saveFilePath_graph="E:\\co-training\\trainW2v\\link\\label_link_vec_id.txt";
          String targetFilePath_text=saveLine2Vec_text;
          String saveFilePath_text="E:\\co-training\\trainW2v\\doc\\label_doc_vec_id.txt";
        InitDBLPClassLabel initDBLPClassLabel_graph=new InitDBLPClassLabel(labelsFilePath,targetFilePath_graph,saveFilePath_graph);
        InitDBLPClassLabel initDBLPClassLabel_text=new InitDBLPClassLabel(labelsFilePath,targetFilePath_text,saveFilePath_text);


        MultiLayerNetwork model_graph=null;
        // model_graph=graphTrainModel.GetNNModel("L_data2",L_list2);
        MultiLayerNetwork model_Text=null;
        //  model_Text=TextTrainModel.GetNNModel("L_data1",L_list1);
        for(int i=1;i<=20;i++) {
            //1、根据label训练模型
            model_graph = graphTrainModel.GetNNModel("L_data2", L_list2);
            model_Text=TextTrainModel.GetNNModel("L_data1",L_list1);
            //2、由得到的模型去给unlabeled集合打分，取置信度高的K
            map_graph = graphChoosePropIdex.getKPropIndex(model_graph, 20);
            map_text = textChoosePropIdex.getKPropIndex(model_Text, 20);
            //3、根据选择添加的K个，对应unlabeled文件中的生成k个添加文件，以备更新word2vec词向量

            System.out.println("************本次迭代选择的大小************"+map_graph.size() +"******************************");

            System.out.println("************本次迭代选择的大小************"+map_text.size()+"********************************");
            String str_addGraph=generateAddFile_graph.updateFile(map_graph);
            String str_addText= generateAddFile_text.updateFile(map_text);

            //4、添加文件后，增量训练，得到新的词向量
//            upTrainW2v_graph.upTrain();
//            upTrainW2v_text.upTrain();
            //5、根据新的词向量，生成新的特征向量
            //FileUtils.write(new File(LabelFilePath_graph),str_addGraph,true);//把添加的文件追加到label文件的最后
            //FileUtils.write(new File(LabelFilePath_text),str_addText,true);
            getUpTrainFeatures_graph.get();
            getUpTrainFeatures_text.get();
            //6、更新类标号
            //添加到Lable集合中的置信度高的，使用真实的标签
            initDBLPClassLabel_graph.replaceLabel();
            initDBLPClassLabel_text.replaceLabel();
            //添加到Lable集合中的置信度高的，使用预测的标签
          //  initDBLPClassLabel_graph.replaceLabel(map_graph,UnlabeledFilePath_graph);
           // initDBLPClassLabel_text.replaceLabel(map_text,UnlabeledFilePath_text);
            getUnlabeledTest_id();
            //7、得到的新的lable复制到文件夹去 U_data2->link
            FileUtils.copyFile(new File(saveFilePath_graph),new File("E:\\co-training\\sample\\deeplearning4j\\textLink\\dblp\\L_data2.csv"));
            FileUtils.copyFile(new File(saveFilePath_text),new File("E:\\co-training\\sample\\deeplearning4j\\textLink\\dblp\\L_data1.csv"));

            //7、再次初始化，NNmodal
            graphTrainModel.init();
            TextTrainModel.init();

            graphChoosePropIdex.init();
            textChoosePropIdex.init();
            L_list1=map2list(map_graph);
            L_list2=map2list(map_text);
        }
        // training();
       // resultMap.put("关系ACC",graphTrainModel.acc);
        resultMap.put("关系F1",graphTrainModel.f1);
       //  resultMap.put("文本ACC",TextTrainModel.acc);
        resultMap.put("文本F1",TextTrainModel.f1);
       // System.out.println("关系ACC："+graphTrainModel.acc);
        System.out.println("关系F1："+graphTrainModel.f1);
        //System.out.println("文本ACC："+TextTrainModel.acc);
        System.out.println("文本F1："+TextTrainModel.f1);

        LineChartsTest fjc = new LineChartsTest("折线图",resultMap);
        fjc.pack();
        RefineryUtilities.centerFrameOnScreen(fjc);
        fjc.setVisible(true);
        List l=new LinkedList<Integer>();
    }
    public static  List<DataSet> map2list(Map<Integer,DataSet> map){
    //public static  List<Integer> map2list(Map<Integer,Integer> map){
        List<DataSet> list=new ArrayList<>();
        //这中遍历是最有效率的遍历
        if(!map.isEmpty()) {
            for (Map.Entry<Integer,DataSet> entry : map.entrySet()) {
                list.add( entry.getValue());
            }
        }else{
            System.out.println("空值传递");
        }

        return list;
    }
    public static void reset() throws Exception{
        String from="E:\\co-training\\sample\\deeplearning4j\\textLink\\";
        String to="E:\\co-training\\sample\\deeplearning4j\\textLink\\dblp\\";
        String[] arr={"L_data1.csv","L_data2.csv","U_data1.csv",
                "U_data2.csv","Test_data1.csv","Test_data2.csv"};
        for(String s:arr){
            FileUtils.copyFile(new File(from+s),new File(to+s));
        }
        String LabelFilePath_graph="E:\\co-training\\trainW2v\\link\\label_link.txt";
        String LabelFilePath_text="E:\\co-training\\trainW2v\\doc\\label_doc.txt";
        FileUtils.copyFile(new File(LabelFilePath_graph),new File(LabelFilePath_graph.replace("txt","temp")));
        FileUtils.copyFile(new File(LabelFilePath_text),new File(LabelFilePath_text.replace("txt","temp")));
    }

    public static void getUnlabeledTest_id() throws IOException{
        String saveW2vPath_graph="E:\\co-training\\trainW2v\\link\\label_link.w2v";
        String saveW2vPath_text="E:\\co-training\\trainW2v\\doc\\label_doc.w2v";
        String unlabel_featureFile_graph="E:\\co-training\\trainW2v\\link\\unlabel_link.txt";
        String unlabel_saveLine2Vec_graph="E:\\co-training\\trainW2v\\link\\unlabel_link_vec.txt";
        String unlabel_featureFile_text="E:\\co-training\\trainW2v\\doc\\unlabel_doc.txt";
        String unlabel_saveLine2Vec_text="E:\\co-training\\trainW2v\\doc\\unlabel_doc_vec.txt";
        String test_featureFile_graph="E:\\co-training\\trainW2v\\link\\test_link.txt";
        String test_saveLine2Vec_graph="E:\\co-training\\trainW2v\\link\\test_link_vec.txt";
        String test_featureFile_text="E:\\co-training\\trainW2v\\doc\\test_doc.txt";
        String test_saveLine2Vec_text="E:\\co-training\\trainW2v\\doc\\test_doc_vec.txt";
        FlattenWord2vec flattenWord2vec_unlabeled_graph=new FlattenWord2vec(saveW2vPath_graph,unlabel_featureFile_graph,unlabel_saveLine2Vec_graph);
        FlattenWord2vec flattenWord2vec_test_graph=new FlattenWord2vec(saveW2vPath_graph,test_featureFile_graph,test_saveLine2Vec_graph);
        FlattenWord2vec flattenWord2vec_unlabeled_text=new FlattenWord2vec(saveW2vPath_text,unlabel_featureFile_text,unlabel_saveLine2Vec_text);
        FlattenWord2vec flattenWord2vec_test_text=new FlattenWord2vec(saveW2vPath_text,test_featureFile_text,test_saveLine2Vec_text);
        flattenWord2vec_unlabeled_graph.flattenAvg();
        flattenWord2vec_test_graph.flattenAvg();
        flattenWord2vec_unlabeled_text.flattenAvg();
        flattenWord2vec_test_text.flattenAvg();

        String labelsFilePath="E:\\co-training\\selected_id_classLabel.txt";
        String targetFilePath_unlabeled_graph=unlabel_saveLine2Vec_graph;
        String saveFilePath_unlabeled_graph="E:\\co-training\\trainW2v\\link\\unlabel_link_vec_id.txt";
        String targetFilePath_unlabeled_text=unlabel_saveLine2Vec_text;
        String saveFilePath_unlabeled_text="E:\\co-training\\trainW2v\\doc\\unlabel_doc_vec_id.txt";
        String targetFilePath_test_graph=test_saveLine2Vec_graph;
        String saveFilePath_test_graph="E:\\co-training\\trainW2v\\link\\test_link_vec_id.txt";
        String targetFilePath_test_text=test_saveLine2Vec_text;
        String saveFilePath_test_text="E:\\co-training\\trainW2v\\doc\\test_doc_vec_id.txt";
        InitDBLPClassLabel initDBLPClassLabel_unlabeled_graph=new InitDBLPClassLabel(labelsFilePath,targetFilePath_unlabeled_graph,saveFilePath_unlabeled_graph);
        InitDBLPClassLabel initDBLPClassLabel_unlabeled_text=new InitDBLPClassLabel(labelsFilePath,targetFilePath_unlabeled_text,saveFilePath_unlabeled_text);
        InitDBLPClassLabel initDBLPClassLabel_test_graph=new InitDBLPClassLabel(labelsFilePath,targetFilePath_test_graph,saveFilePath_test_graph);
        InitDBLPClassLabel initDBLPClassLabel_test_text=new InitDBLPClassLabel(labelsFilePath,targetFilePath_test_text,saveFilePath_test_text);
        initDBLPClassLabel_unlabeled_graph.replaceLabel();
        initDBLPClassLabel_unlabeled_text.replaceLabel();
        initDBLPClassLabel_test_graph.replaceLabel();
        initDBLPClassLabel_test_text.replaceLabel();

        FileUtils.copyFile(new File(saveFilePath_unlabeled_graph),new File("E:\\co-training\\sample\\deeplearning4j\\textLink\\dblp\\U_data2.csv"));
        FileUtils.copyFile(new File(saveFilePath_unlabeled_text),new File("E:\\co-training\\sample\\deeplearning4j\\textLink\\dblp\\U_data1.csv"));

        FileUtils.copyFile(new File(saveFilePath_test_graph),new File("E:\\co-training\\sample\\deeplearning4j\\textLink\\dblp\\Test_data2.csv"));
        FileUtils.copyFile(new File(saveFilePath_test_text),new File("E:\\co-training\\sample\\deeplearning4j\\textLink\\dblp\\Test_data1.csv"));



    }
}
