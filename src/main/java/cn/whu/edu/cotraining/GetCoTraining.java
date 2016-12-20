package cn.whu.edu.cotraining;

import org.deeplearning4j.examples.bczhang.LineChartsTest;
import org.deeplearning4j.examples.bczhang.NNconf;
import org.deeplearning4j.examples.bczhang.NewChoosePropIdex;
import org.deeplearning4j.examples.bczhang.NewTrainModel;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.jfree.ui.RefineryUtilities;
import org.nd4j.linalg.dataset.DataSet;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by bczhang on 2016/12/12.
 */
public class GetCoTraining {
    public static void main(String[]args)throws Exception{
        Map<String,List<Double>> resultMap=new HashMap<>();
        //  训练model
        GetModel graphTrainModel=new GetModel("L_data2","Test_data2");
        graphTrainModel.init();
        GetModel TextTrainModel=new GetModel("L_data1","Test_data1");
        TextTrainModel.init();
        // K置信度选择器
        SelectKPropIdex graphChoosePropIdex=new SelectKPropIdex("U_data2","U_data1");
        SelectKPropIdex textChoosePropIdex=new SelectKPropIdex("U_data1","U_data2");
        graphChoosePropIdex.init();
        textChoosePropIdex.init();
        //  设置不同的配置文件


        List<DataSet> L_list1=null;
        List<DataSet> L_list2=null;
        MultiLayerNetwork model_graph=null;
       // model_graph=graphTrainModel.GetNNModel("L_data2",L_list2);
        MultiLayerNetwork model_Text=null;
      //  model_Text=TextTrainModel.GetNNModel("L_data1",L_list1);
        for(int i=1;i<=50;i++) {
            model_graph = graphTrainModel.GetNNModel("L_data2", L_list2);
            model_Text=TextTrainModel.GetNNModel("L_data1",L_list1);
            L_list1 = graphChoosePropIdex.getKPropIndex(model_graph, 4);
            L_list2 = textChoosePropIdex.getKPropIndex(model_Text, 4);

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
