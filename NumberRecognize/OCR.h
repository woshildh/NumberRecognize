#pragma once
#include"stdafx.h"
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\ml.hpp>
#include<iostream>
using namespace std;
using namespace cv;
using namespace cv::ml;
class OCR {
	private:
		//这五个变量都提供了相应的函数可以进行修改
		int k;//训练时的近邻个数
		int size;//设置图像的大小
		char file_path[200]; //这里指的是预处理后的数据的路径
		char pre_file_path[200]; //这里指的是预处理前数据的路径
		int trainNum; //设置训练数目,指的是每个类中训练样本数目


		Mat data;//传入训练的数据
		Mat labels;//传入训练的标签
		Mat test_data;//传入测试的数据
		Mat test_labels;//传入测试的标签
		Ptr<KNearest> model = KNearest::create(); //KNN训练模型
		Ptr<TrainData> trainData;

	public:
		OCR();
		void setK(int num);
		void setSize(int num);
		void setFilePath(char *path);
		void setPreFilePath(char *path);
		void setTrainNum(int num);

		void read_data();
		void img_proc(int user_size); //对数据进行预处理
		void train(); //对数据进行训练
		float predict(Mat testData); //对数据进行预测
		float predict_pre(Mat pre_img);//对原始图像进行预测
		float test(); //测试 测试数据
 };
