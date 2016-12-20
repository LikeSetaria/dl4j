package cn.whu.edu.cotraining;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.FileRecordReader;
import org.datavec.api.records.reader.impl.misc.LibSvmRecordReader;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.records.writer.impl.FileRecordWriter;
import org.datavec.api.records.writer.impl.misc.LibSvmRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by bczhang on 2016/12/9.
 */
public class Test {
    public static void main(String[]args) throws Exception{
//        Configuration conf=new Configuration();
//        conf.set(LineRecordReader.APPEND_LABEL,"true");
//        conf.set(LibSvmRecordReader.NUM_FEATURES,"100");
//        LibSvmRecordReader libSVMRR = new LibSvmRecordReader();
//        libSVMRR.initialize(conf,new FileSplit(new File("E:\\co-training\\source\\textLink\\DBLP\\doc_vec_text.txt")));
//        DataSetIterator trainIter = new RecordReaderDataSetIterator(libSVMRR,50,0,2);
//

        Configuration conf = new Configuration();
        conf.set(FileRecordReader.APPEND_LABEL, "true");
        File out = new File("iris.libsvm.out");
        if(out.exists()) out.delete();
        conf.set(FileRecordWriter.PATH, out.getAbsolutePath());
        RecordReader libSvmRecordReader = new LibSvmRecordReader();
        libSvmRecordReader.initialize(conf, new FileSplit(new File("E:\\co-training\\source\\textLink\\DBLP\\iris.libsvm")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(libSvmRecordReader,50,1,3);

        //dataSets.getLabels();
        while(trainIter.hasNext()) {
            DataSet dataSets = trainIter.next();
            dataSets.getLabels();
            System.out.print(dataSets.getLabels());
        }

        RecordWriter writer = new LibSvmRecordWriter();
        writer.setConf(conf);
        List<List<Writable>> data = new ArrayList<>();
        while (libSvmRecordReader.hasNext()) {
            List<Writable> record = libSvmRecordReader.next();
            writer.write(record);
            data.add(record);
           // System.out.print(record);
        }
        writer.close();

        out.deleteOnExit();
        List<List<Writable>> test = new ArrayList<>();
        RecordReader testLibSvmRecordReader = new LibSvmRecordReader();
        testLibSvmRecordReader.initialize(conf, new FileSplit(out));
        while (testLibSvmRecordReader.hasNext())
            test.add(testLibSvmRecordReader.next());
        //assertEquals(data, test);




    }
}
