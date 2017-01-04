package cn.whu.edu.multiCoTraining;

import org.deeplearning4j.examples.bczhang.LineChartsTest;
import org.deeplearning4j.examples.bczhang.NNconf;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.jfree.ui.RefineryUtilities;
import org.nd4j.linalg.dataset.DataSet;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by bczhang on 2016/12/20.
 */
public class Main {
    public static  void  main(String[] agrs) throws Exception{
        classicCoT();
       // mergeCoT();
    }
    public static void mergeCoT()throws  Exception{
        Map<String,List<Double>> resultMap=new HashMap<>();
        //  训练model
        NNModel graphTrainModel=new NNModel("L_data2","Test_data2");
        graphTrainModel.init();
        NNModel TextTrainModel=new NNModel("L_data1","Test_data1");
        TextTrainModel.init();
        // K置信度选择器
        TopConfidence graphChoosePropIdex=new TopConfidence("U_data2","U_data1");
        //TopConfidence textChoosePropIdex=new TopConfidence("U_data1","U_data2");
        graphChoosePropIdex.init();
        // textChoosePropIdex.init();
        //  设置不同的配置文件
        NNconf graphnnconf=new NNconf(12);

        NNconf textnnconf=new NNconf(16);

        MultiLayerConfiguration graphconf=graphnnconf.getConf();
        MultiLayerConfiguration textconf=textnnconf.getConf();
        List<DataSet> L_list1=null;
        List<DataSet> L_list2=null;
        MultiLayerNetwork model_graph=new MultiLayerNetwork(graphconf);
        model_graph=graphTrainModel.GetNNModel("L_data2",L_list2);
        MultiLayerNetwork model_Text=new MultiLayerNetwork(textconf);
        model_Text=TextTrainModel.GetNNModel("L_data1",L_list1);
        Map<String,List<DataSet>> KPropMaps=null;
        for(int i=1;i<=20;i++) {
            KPropMaps=graphChoosePropIdex.getKPropIndex(model_graph,model_Text,30);
            L_list1 = KPropMaps.get("text");
            L_list2 =KPropMaps.get("relation");
            model_graph = graphTrainModel.GetNNModel("L_data2", L_list2);
            model_Text=TextTrainModel.GetNNModel("L_data1",L_list1);
        }
        // training();
     //   resultMap.put("关系ACC",graphTrainModel.acc);
        resultMap.put("关系F1",graphTrainModel.f1);
      //  resultMap.put("文本ACC",TextTrainModel.acc);
        resultMap.put("文本F1",TextTrainModel.f1);
        //System.out.println("关系ACC："+graphTrainModel.acc);
        System.out.println("关系F1："+graphTrainModel.f1);
        //System.out.println("文本ACC："+TextTrainModel.acc);
        System.out.println("文本F1："+TextTrainModel.f1);

        LineChartsTest fjc = new LineChartsTest("折线图",resultMap);
        fjc.pack();
        RefineryUtilities.centerFrameOnScreen(fjc);
        fjc.setVisible(true);
    }
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
        MultiLayerNetwork model_graph=null;
        // model_graph=graphTrainModel.GetNNModel("L_data2",L_list2);
        MultiLayerNetwork model_Text=null;
        //  model_Text=TextTrainModel.GetNNModel("L_data1",L_list1);
        for(int i=1;i<=20;i++) {
            model_graph = graphTrainModel.GetNNModel("L_data2", L_list2);
            model_Text=TextTrainModel.GetNNModel("L_data1",L_list1);
            L_list1 = graphChoosePropIdex.getKPropIndex(model_graph, 20);
            L_list2 = textChoosePropIdex.getKPropIndex(model_Text, 20);

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
    }
}
