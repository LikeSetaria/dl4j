package org.deeplearning4j.examples.nlp.word2vec;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.LowCasePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.ui.UiServer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Collection;

/**
 * Created by agibsonccc on 10/9/14.
 *
 * Neural net that processes text into wordvectors. See below url for an in-depth explanation.
 * https://deeplearning4j.org/word2vec.html
 */
public class Word2VecRawTextExample {

    private static Logger log = LoggerFactory.getLogger(Word2VecRawTextExample.class);

    public static void main(String[] args) throws Exception {

        // Gets Path to Text file
        String filePath = new ClassPathResource("cora.collection").getFile().getAbsolutePath();
        filePath="E:\\co-training\\sample\\deeplearning4j\\textLink\\dblp_coTraining2vec\\label_doc.txt";
        String savePath="E:\\co-training\\sample\\deeplearning4j\\textLink\\dblp_coTraining2vec\\label_doc2vec.w2v";
        log.info("Load & Vectorize Sentences....");

        word2vecExample(filePath,savePath);
//        SentenceIterator iter = new BasicLineIterator(filePath);
//        TokenizerFactory t = new DefaultTokenizerFactory();
//        t.setTokenPreProcessor(new CommonPreprocessor());
//        CommonPreprocessor c=new CommonPreprocessor();
//        c.preProcess("");
//        log.info("Building model....");
//        Word2Vec vec = new Word2Vec.Builder()
//                .minWordFrequency(5)
//                .iterations(1)
//                .layerSize(100)
//                .seed(42)
//                .windowSize(5)
//                .iterate(iter)
//               // .tokenizerFactory(t)
//                .build();
//        log.info("Fitting Word2Vec model....");
//        vec.fit();
//        log.info("Writing word vectors to text file....");
//
//        // Write word vectors to file
//        WordVectorSerializer.writeWordVectors(vec, savePath);


    }

    public static void word2vecExample(String filepath,String savePath) throws IOException{
        log.info("Load data....");
        SentenceIterator iter = new LineSentenceIterator(new File(filepath));
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });
        // 在每行中按空格分词
        TokenizerFactory t = new DefaultTokenizerFactory();

        t.setTokenPreProcessor(new LowCasePreProcessor());
        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
            .minWordFrequency(1)
            .iterations(1)
            .layerSize(100)
            .seed(42)
            .windowSize(5)
            .iterate(iter)
            .tokenizerFactory(t)
            .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();
        WordVectorSerializer.writeWordVectors(vec, savePath);
        log.info("Closest Words:");


    }
}
