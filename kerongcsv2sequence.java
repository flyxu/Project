package xuxiang.xufly.maventest;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.io.Text;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class kerongcsv2sequence {
	public static void main(String[] args) throws IOException {
		Configuration conf = new Configuration();//获取环境变量  
        FileSystem fs = FileSystem.get(conf);//获取文件系统  
        Path path = new Path("/home/hadoop/kerong/traindata-seq");//定义路径  
        BufferedReader reader = new BufferedReader(new FileReader("/home/hadoop/kerong/traindata.csv"));
        SequenceFile.Writer writer = new Writer(fs, conf,path, Text.class, VectorWritable.class);
        String line=reader.readLine();
        while((line=reader.readLine())!= null){
        	   String[] c = line.split(",");
        	   Text key=new Text();
        	   VectorWritable vector=new VectorWritable();
        	   Vector vec = new RandomAccessSparseVector(c.length-1);
        	   key.set("/"+c[c.length-1]+"/");
        	   double  d[]=new double[c.length-1];
        	   for(int i=0;i<c.length-1;i++){
        		   d[i]=Double.parseDouble(c[i]);
        	   }
        	   vec.assign(d );
        	   vector.set(vec);
        	   writer.append(key,vector);
        }
        writer.close();
        reader.close();
        
	}
}
