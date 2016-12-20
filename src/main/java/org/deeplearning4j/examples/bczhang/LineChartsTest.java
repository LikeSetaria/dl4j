package org.deeplearning4j.examples.bczhang;

import javax.swing.JPanel;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.*;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.category.LineAndShapeRenderer;
import org.jfree.chart.title.LegendTitle;
import org.jfree.chart.title.TextTitle;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

import java.awt.*;
import java.util.List;
import java.util.Map;

public class LineChartsTest extends ApplicationFrame {
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
        JFreeChart jfreechart = createChart2(createDataset(map));
        return new ChartPanel(jfreechart);
    }// 生成图表主对象JFreeChart

    /**
     * 根据PieDataset创建JFreeChart对象
     * @return JFreeChart
     */
    public static JFreeChart createChart2(CategoryDataset categoryDataset) {
        //JFreeChart类是一个制图对象类，先用它来创建一个制图对象chart
        //ChartFactory类是制图工厂类，用它来为制图对象chart完成实例化
        //createLineChart()是制图工厂的一个方法，用来创建一个常规的折线图对象
        JFreeChart chart = ChartFactory.createLineChart(
            "cotraining&&NN",                 //图表标题
            "Epochs",                        //X轴标题
            "%",                        //Y轴标题
            categoryDataset,              //数据集
            PlotOrientation.VERTICAL,     //绘制方向
            true,                         //是否显示图例
            false,                        //是否采用标准生成器
            false                         //是否支持超链接
        );
        //通过JFreeChart对象的 setTitle方法，修改统计图表的标题部分（包括修改图表标题内容、字体大小等）
        chart.setTitle(new TextTitle("cotraining&&NN", new Font("黑体", Font.ITALIC , 22)));
        //调用 JFreeChart对象的 getLegend(int index)方法，取得该图表的指定索引的图例对象，通过 LegendTitle对象来修改统计图表的图例
        LegendTitle legend = chart.getLegend(0);
        //设置图例的字体和字体大小，即位于下方的字的字体和大小
        legend.setItemFont(new Font("宋体", Font.BOLD, 14));
        // 设置画布背景色
        chart.setBackgroundPaint(new Color(192, 228, 106));
        //取得折线图的绘图(plot)对象
        CategoryPlot plot = chart.getCategoryPlot();
        //设置数据区的背景透明度，范围在0.0～1.0间
        plot.setBackgroundAlpha(0.5f);
        // 设置数据区的前景透明度，范围在0.0～1.0间
        plot.setForegroundAlpha(0.5f);
        // 设置横轴字体
        plot.getDomainAxis().setLabelFont(new Font("黑体", Font.BOLD, 14));
        // 设置坐标轴标尺值字体
        plot.getDomainAxis().setTickLabelFont(new Font("宋体", Font.BOLD, 12));
        // 设置纵轴字体
        plot.getRangeAxis().setLabelFont(new Font("黑体", Font.BOLD, 14));
        // 设置绘图区背景色
        plot.setBackgroundPaint(Color.WHITE);
        // 设置水平方向背景线颜色
        plot.setRangeGridlinePaint(Color.BLACK);
        // 设置是否显示水平方向背景线,默认值为true
        plot.setRangeGridlinesVisible(true);
        // 设置垂直方向背景线颜色
        plot.setDomainGridlinePaint(Color.BLACK);
        // 设置是否显示垂直方向背景线,默认值为false
        plot.setDomainGridlinesVisible(true);
        // 没有数据时显示的消息
        plot.setNoDataMessage("没有相关统计数据");
        plot.setNoDataMessageFont(new Font("黑体", Font.CENTER_BASELINE, 16));
        plot.setNoDataMessagePaint(Color.RED);
        // 获取折线对象
        LineAndShapeRenderer renderer = (LineAndShapeRenderer) plot
            .getRenderer();
        //设置折点处以某种形状凸出
        renderer.setShapesVisible(true);
        renderer.setDrawOutlines(true);
        renderer.setUseFillPaint(true);
        renderer.setFillPaint(java.awt.Color.WHITE);
        //设置显示折点处的数据值
        //renderer.setBaseItemLabelGenerator (new StandardCategoryItemLabelGenerator ());
        //renderer.setItemLabelFont (new Font ("黑体", Font.PLAIN, 12));
        //renderer.setItemLabelsVisible (true);
        BasicStroke realLine = new BasicStroke(2.0f); // 设置实线
        float dashes[] = { 8.0f }; // 定义虚线数组
        BasicStroke brokenLine = new BasicStroke(2.0f, // 线条粗细
            BasicStroke.CAP_SQUARE, // 端点风格
            BasicStroke.JOIN_MITER, // 折点风格
            8.f, // 折点处理办法
            dashes, // 虚线数组
            0.0f); // 虚线偏移量
        // 利用虚线绘制
        renderer.setSeriesStroke(0, brokenLine);
        // 利用虚线绘制
        renderer.setSeriesStroke(1, brokenLine);
        // 利用实线绘制
        renderer.setSeriesStroke(2, realLine);
        // 利用实线绘制
        renderer.setSeriesStroke(3, realLine);
        //设置折线的颜色
        renderer.setSeriesPaint(0, Color.BLACK);
        renderer.setSeriesPaint(1, Color.RED);
        renderer.setSeriesPaint(2, Color.BLUE);
        renderer.setSeriesPaint(3, Color.MAGENTA);
        return chart;
    }
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
