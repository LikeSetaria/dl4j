package org.deeplearning4j.examples.bczhang;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by bczhang on 2016/11/27.
 */
public class NNconf {
    public NNconf(){
        this.conf=conf;
    }
    int seed = 123;
    double learningRate = 0.01;
    int batchSize = 50;
    int nEpochs = 30;

    int numInputs = 12;
    int numOutputs = 2;
    int numHiddenNodes = 350;
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(seed)
        .iterations(1)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .learningRate(learningRate)
        .updater(Updater.NESTEROVS).momentum(0.9)
        .list()
        .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
            .weightInit(WeightInit.XAVIER)
            .activation("relu")
            .build())
        .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .weightInit(WeightInit.XAVIER)
            .activation("softmax").weightInit(WeightInit.XAVIER)
            .nIn(numHiddenNodes).nOut(numOutputs).build())
        .pretrain(false).backprop(true).build();
}
