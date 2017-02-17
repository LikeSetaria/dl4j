package cn.whu.edu.multiCoTraining;

import org.deeplearning4j.examples.nlp.word2vec.Word2VecRawTextExample;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.util.Collection;

/**
 * 增量训练词向量，词向量训练好以后保存参数，以后添加部分词，继续训练加快速度
 * Created by 宝超 on 2017/2/14.
 */
public class IncrementalTrainWord2vec {
    private static Logger log = LoggerFactory.getLogger(Word2VecRawTextExample.class);
    String initFilePath="E:\\co-training\\label_doc.txt";
    String addFilePath="E:\\co-training\\label_doc_add.txt";
    String saveModelPath="E:\\co-training\\label_doc.w2vModel";
    String saveW2vPath="E:\\co-training\\label_doc_vec.txt";
    public static void main(String[] args) throws Exception{

        String initFilePath="E:\\co-training\\label_doc.txt";
        String addFilePath="E:\\co-training\\label_doc_add.txt";
        String saveModelPath="E:\\co-training\\label_doc.w2vModel";
        String saveW2vPath="E:\\co-training\\label_doc_vec.txt";
        IncrementalTrainWord2vec increTrain=new IncrementalTrainWord2vec(initFilePath,addFilePath,saveW2vPath,saveModelPath);
        increTrain.upTrain();
    }
    public IncrementalTrainWord2vec(String initFilePath,String addFilePath,String saveW2vPath,String saveModelPath){
        this.initFilePath=initFilePath;
        this.addFilePath=addFilePath;
        this.saveModelPath=saveModelPath;
        this.saveW2vPath=saveW2vPath;


    }

    /**
     *更新w2v模型
     * @initpath 初始化训练w2v文件
     * @savepath 保存w2v词向量
     * @addPath 更新w2v model添加文件
     * @saveModelpath 保存最新模型参数
     */
    public void upTrain()throws Exception{
        File modelFile=new File(saveModelPath);
        //如果model文件不存在，则初次生成文件。否则加载model
        if(!modelFile.exists()) {
            log.info("word2vec不存在，初始化");
            SentenceIterator iter = new BasicLineIterator(initFilePath);
            // Split on white spaces in the line to get words
            TokenizerFactory t = new DefaultTokenizerFactory();
            t.setTokenPreProcessor(new CommonPreprocessor());
            // manual creation of VocabCache and WeightLookupTable usually isn't necessary
            // but in this case we'll need them
            InMemoryLookupCache cache = new InMemoryLookupCache();
            WeightLookupTable<VocabWord> table = new InMemoryLookupTable.Builder<VocabWord>()
                    .vectorLength(100)
                    .useAdaGrad(false)
                    .cache(cache)
                    .lr(0.025f).build();
            log.info("Building model....");
            Word2Vec vec = new Word2Vec.Builder()
                    .minWordFrequency(1)//最小词频设置位1，一般设置为5，如果为5词频小于5的将不会输出向量
                    .iterations(1)
                    .epochs(1)
                    .layerSize(100)
                    .seed(42)
                    .windowSize(5)
                    .iterate(iter)
                    .tokenizerFactory(t)
                    .lookupTable(table)
                    .vocabCache(cache)
                    .build();
            log.info("Fitting Word2Vec model....");
            vec.fit();



        log.info("Closest Words:");
        Collection<String> lst = vec.wordsNearest("a", 10);
        System.out.println("10 Words closest to 'SIZE': " + lst);
            //
            WordVectorSerializer.writeFullModel(vec, saveModelPath);
            WordVectorSerializer.writeWordVectors(vec, saveW2vPath);
            File addFile=new File(addFilePath);
            log.info("更新model及语料文件存在，更新model.....");
            if(addFile.exists()){
                Word2Vec word2Vec = WordVectorSerializer.loadFullModel(saveModelPath);
                SentenceIterator iterator = new BasicLineIterator(addFilePath);
                TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
                tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

                word2Vec.setTokenizerFactory(tokenizerFactory);
                word2Vec.setSentenceIter(iterator);
                //train w2v
                word2Vec.fit();

                //保存更新后的w2v model 以及新的词向量
                WordVectorSerializer.writeFullModel(word2Vec, saveModelPath);
                WordVectorSerializer.writeWordVectors(word2Vec, saveW2vPath);
            }
        }

        else {
            log.info("moedl文件存在，根据添加语料更新model");
            Word2Vec word2Vec = WordVectorSerializer.loadFullModel(saveModelPath);
            File addFile=new File(addFilePath);
            if(addFile.exists()) {
                SentenceIterator iterator = new BasicLineIterator(addFilePath);
                TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
                tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

                word2Vec.setTokenizerFactory(tokenizerFactory);
                word2Vec.setSentenceIter(iterator);
                //train w2v
                word2Vec.fit();

                //保存更新后的w2v model 以及新的词向量
                WordVectorSerializer.writeFullModel(word2Vec, saveModelPath);
                WordVectorSerializer.writeWordVectors(word2Vec, saveW2vPath);
            }
            else {
                log.info("渐增文件不存在，model没有更新");
            }
        }

       // Word2Vec word2Vec = WordVectorSerializer.loadFullModel("pathToSaveModel.txt");
    }
}
