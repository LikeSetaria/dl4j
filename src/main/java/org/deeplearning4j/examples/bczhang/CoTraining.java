package org.deeplearning4j.examples.bczhang;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/**
 * Created by bczhang on 2016/11/26.
 */
public class CoTraining {
    public static void main(String[]args)throws Exception{
        TrainModel trainmodel=new TrainModel();
        NNconf nnconf=new NNconf();
        MultiLayerConfiguration conf=nnconf.conf;
        ChoosePropIdex choosePropIdex=new ChoosePropIdex();
        MultiLayerNetwork model_graph=new MultiLayerNetwork(conf);
        model_graph=trainmodel.GetNNModel(conf);
        //MultiLayerNetwork model_Text=new MultiLayerNetwork(conf);
        //model_Text=trainmodel.GetNNModel(conf);
        choosePropIdex.getKPropIndex(model_graph,21);

    }
}
