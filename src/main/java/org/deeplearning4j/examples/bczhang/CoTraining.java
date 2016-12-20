package org.deeplearning4j.examples.bczhang;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.jfree.ui.RefineryUtilities;
import org.nd4j.linalg.dataset.DataSet;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by bczhang on 2016/11/26.
 */
public class CoTraining {
    public static void main(String[]args)throws Exception{
        Map<String,List<Double>> resultMap=new HashMap<>();
      //  训练model
        NewTrainModel graphTrainModel=new NewTrainModel("L_data2","Test_data2");
        graphTrainModel.init();
        NewTrainModel TextTrainModel=new NewTrainModel("L_data1","Test_data1");
        TextTrainModel.init();
       // K置信度选择器
        NewChoosePropIdex graphChoosePropIdex=new NewChoosePropIdex("U_data2","U_data1");
        NewChoosePropIdex textChoosePropIdex=new NewChoosePropIdex("U_data1","U_data2");
        graphChoosePropIdex.init();
        textChoosePropIdex.init();
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
        for(int i=1;i<=50;i++) {
            L_list1 = graphChoosePropIdex.getKPropIndex(model_graph, 1);
            L_list2 = textChoosePropIdex.getKPropIndex(model_Text, 1);
            model_graph = graphTrainModel.GetNNModel("L_data2", L_list2);
            model_Text=TextTrainModel.GetNNModel("L_data1",L_list1);
        }
       // training();
        resultMap.put("关系ACC",graphTrainModel.acc);
        resultMap.put("关系F1",graphTrainModel.f1);
        resultMap.put("文本ACC",TextTrainModel.acc);
        resultMap.put("文本F1",TextTrainModel.f1);
        System.out.println("关系ACC："+graphTrainModel.acc);
        System.out.println("关系F1："+graphTrainModel.f1);
        System.out.println("文本ACC："+TextTrainModel.acc);
        System.out.println("文本F1："+TextTrainModel.f1);

        LineChartsTest fjc = new LineChartsTest("折线图",resultMap);
        fjc.pack();
        RefineryUtilities.centerFrameOnScreen(fjc);
        fjc.setVisible(true);
        //training2();
    }

    /**
     * 选择累加的量时，进行交集选择
     * @throws Exception
     */
    public static  void training2()throws  Exception{

        Map<String,List<Double>> resultMap=new HashMap<>();
        //  训练model
        NewTrainModel graphTrainModel=new NewTrainModel("L_data2","Test_data2");
        graphTrainModel.init();
        NewTrainModel TextTrainModel=new NewTrainModel("L_data1","Test_data1");
        TextTrainModel.init();
        // K置信度选择器
        NewChoosePropIdex graphChoosePropIdex=new NewChoosePropIdex("U_data2","U_data1");
        //NewChoosePropIdex textChoosePropIdex=new NewChoosePropIdex("U_data1","U_data2");
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
        for(int i=1;i<=50;i++) {
            KPropMaps=graphChoosePropIdex.getKPropIndex(model_graph,model_Text,1);
            L_list1 = KPropMaps.get("text");
            L_list2 =KPropMaps.get("relation");
            model_graph = graphTrainModel.GetNNModel("L_data2", L_list2);
            model_Text=TextTrainModel.GetNNModel("L_data1",L_list1);
        }
        // training();
        resultMap.put("关系ACC",graphTrainModel.acc);
        resultMap.put("关系F1",graphTrainModel.f1);
        resultMap.put("文本ACC",TextTrainModel.acc);
        resultMap.put("文本F1",TextTrainModel.f1);
        System.out.println("关系ACC："+graphTrainModel.acc);
        System.out.println("关系F1："+graphTrainModel.f1);
        System.out.println("文本ACC："+TextTrainModel.acc);
        System.out.println("文本F1："+TextTrainModel.f1);

        LineChartsTest fjc = new LineChartsTest("折线图",resultMap);
        fjc.pack();
        RefineryUtilities.centerFrameOnScreen(fjc);
        fjc.setVisible(true);
}
}
