package org.deeplearning4j.examples.bczhang;

import javax.swing.JPanel;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.*;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

import java.util.List;
import java.util.Map;

class LineChartsTest extends ApplicationFrame {
    /** * */
    private static final long serialVersionUID = 1L;

    public LineChartsTest(String s,Map<String,List<Double>> map) {
        super(s);
        setContentPane(createDemoLine(map));
    }

    public static void main(String[] args) {
        Map<String,List<Double>> map=null;
        LineChartsTest fjc = new LineChartsTest("折线图",map);
        fjc.pack();
        RefineryUtilities.centerFrameOnScreen(fjc);
        fjc.setVisible(true);

    }// 生成显示图表的面板

    public static JPanel createDemoLine(Map<String,List<Double>> map) {
        JFreeChart jfreechart = createChart(createDataset(map));
        return new ChartPanel(jfreechart);
    }// 生成图表主对象JFreeChart

    public static JFreeChart createChart(DefaultCategoryDataset linedataset) { // 定义图表对象
        JFreeChart chart = ChartFactory.createLineChart("co-training&&NN", // 折线图名称
            "Iteration number", // 横坐标名称
            "%", // 纵坐标名称
            linedataset, // 数据
            PlotOrientation.VERTICAL, // 水平显示图像
            true, // include legend
            true, // tooltips
            false // urls
        );
        CategoryPlot plot = chart.getCategoryPlot();
        plot.setRangeGridlinesVisible(true); // 是否显示格子线
        plot.setBackgroundAlpha(0.3f); // 设置背景透明度
        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
//        rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
//        rangeAxis.setAutoRangeIncludesZero(true);
//        rangeAxis.setUpperMargin(0.20);
//        rangeAxis.setLabelAngle(Math.PI / 2.0);
        rangeAxis .setAutoTickUnitSelection(false);
        double unit=0.05;//刻度的长度
        NumberTickUnit ntu= new NumberTickUnit(unit);
        rangeAxis .setTickUnit(ntu);



        return chart;
    }// 生成数据

    public static DefaultCategoryDataset createDataset(Map<String,List<Double>> map) {
        DefaultCategoryDataset linedataset = new DefaultCategoryDataset();
        // 各曲线名称
        String series1 = "text F1";
        String series2 = "text ACC";
        String series3 = "relation F1";
        String series4 = "relation ACC";
        // 横轴名称(列名称)
//        String type1 = "Text";
//        String type2 = "2月";
//        String type3 = "3月";
//        String type4 = "4月";

        for(Map.Entry<String,List<Double>> entry:map.entrySet()){
            String key=entry.getKey();
            List<Double> value=entry.getValue();
            if(key.equals("关系ACC")){
                Integer i=0;
                for(Double d:value){
                    linedataset.addValue(d,series4,i);
                    i++;
                }
            }
           else if(key.equals("关系F1")){
                Integer i=0;
                for(Double d:value){
                    linedataset.addValue(d,series3,i);
                    i++;
                }
            }
           else if(key.equals("文本ACC")){
                Integer i=0;
                for(Double d:value){
                    linedataset.addValue(d,series2,i);
                    i++;
                }
            }
           else if(key.equals("文本F1")){
                Integer i=0;
                for(Double d:value){
                    linedataset.addValue(d,series1,i);
                    i++;
                }
            }

        }
//        linedataset.addValue(0.0, series1, type1);
//        linedataset.addValue(4.2, series1, type2);
//        linedataset.addValue(3.9, series1, type3);
//        linedataset.addValue(3.9, series1, type4);
//
//        linedataset.addValue(1.0, series2, type1);
//        linedataset.addValue(5.2, series2, type2);
//        linedataset.addValue(7.9, series2, type3);
//        linedataset.addValue(3.9, series2, type4);
//
//        linedataset.addValue(2.0, series3, type1);
//        linedataset.addValue(9.2, series3, type2);
//        linedataset.addValue(8.9, series3, type3);
//        linedataset.addValue(3.9, series3, type4);
//
//        linedataset.addValue(3.9, series4, type1);
//        linedataset.addValue(3.4, series4, type2);
//        linedataset.addValue(1.9, series4, type3);
//        linedataset.addValue(7.9, series4, type4);
        return linedataset;
    }
}
