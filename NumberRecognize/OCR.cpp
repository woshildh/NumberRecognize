/*本文件实现 OCR.hpp中定义的类的函数  */
#include"stdafx.h"
#include"preprocess.hpp"
#include"OCR.h"
#include<opencv2\ml\ml.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<highgui\highgui.hpp>
#include<cstring>
#include<iostream>
using namespace std;
using namespace cv;
using namespace cv::ml;
OCR::OCR()
{
	k = 5;
	size = 64;
	strcpy(file_path, "pro_data/");
	strcpy(pre_file_path, "pre_data/");
	trainNum = 70;
}

void OCR::setFilePath(char *path)
{
	strcpy(file_path, path);
}
void OCR::setK(int num)
{
	k = num;
}
void OCR::setPreFilePath(char *path)
{
	strcpy(pre_file_path, path);
}
void OCR::setSize(int num)
{
	size = num;
}
void OCR::setTrainNum(int num)
{
	this->trainNum = num;
	cout << "trainNum:" << trainNum << endl;
}

void OCR::read_data() //从file_path中读入数据，并且存入data和labels中
{
	char path[200];
	int num = 0;     //记录有多少张图片在读入时出错了
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < trainNum; j++)
		{
			sprintf(path, "%s%d/%03d%s", file_path, i, j+100*i,".pbm");
			Mat temp = imread(path,0);
			if (!temp.data)
			{
				num++; continue;
			}
			data.push_back(temp.reshape(0,1));
			labels.push_back(i);
		}
	}
	data.convertTo(data,CV_32F);
	labels.convertTo(labels,CV_32F);
	trainData = TrainData::create(data, ROW_SAMPLE, labels);
	cout << "数据矩阵的大小为" << data.rows << " * " << data.cols << endl;
	cout << "标签矩阵的 大小为" << labels.rows << "*" << labels.cols << endl;
	cout << "共有" << num << "张图片读入出错了..." <<endl;
}

void OCR::img_proc(int user_size) //对数据进行预处理，将图片尺寸压缩为一定的比例
{
	if (user_size == size)
		return;
	size = user_size;
	compress(pre_file_path, file_path, user_size, user_size);
}
void OCR::train() //对数据进行训练
{
	
	//对模型初始化
	model->setDefaultK(k); model->setIsClassifier(true);
	//训练模型
	model->train(trainData);
}
float OCR::predict(Mat testData) //对数据进行预测,testData应该是一个向量
{
	return model->predict(testData);
}
float OCR::predict_pre(Mat pre_img)
{
	float res;
	Mat pro_img;//处理后的图像
	Rect border;//内容的边界
	findBorder(pre_img, border);
	preprocess(pre_img, border, pro_img, size, size);
	pro_img = pro_img.reshape(0, 1);
	pro_img.convertTo(pro_img, CV_32F);//转成CV_32F
	res=model->predict(pro_img);
	return res;
}
float OCR::test()//测试 测试数据
{
	//生成测试集矩阵和测试集标签
	char path[200];
	float num = 0.f; //num表示总共有多少个测试数据
	float positive_num = 0.f;//表示预测正确的数目
	for (int i = 0; i < 10; i++)
	{
		for (int j = trainNum; j<100;j++)
		{
			//cout << "------------" << endl;
			sprintf(path, "%s%d/%03d%s", file_path, i, j+i*100,".pbm");
			//cout << path << endl;//测试时所使用
			Mat temp = imread(path, 0);
			if (!temp.data)
			{
				continue;
			}
			//cout << "-------------***-------------"<<endl;	
			test_data.push_back(temp.reshape(0,1));
			test_labels.push_back(i);
			num=num+1;
		}
	}
	/*cout << test_data.rows << " " << test_data.cols << endl;
	cout << test_labels.rows << endl;*/
	test_data.convertTo(test_data,CV_32F);
	//test_labels.convertTo(test_labels,CV_32F);  这行代码不能要，否则会出错
	for (int i = 0; i < test_data.rows; i++)
	{
		float res = model->predict(test_data.row(i));
		//cout << res << endl;
		float r = cv::abs(res -(float)test_labels.at<int>(i));
		r = (r <= FLT_EPSILON) ? 1.f: 0.0f;
		positive_num =positive_num+r;
	}
	 float accuracy = positive_num / num;
	//cout << num << "  " << positive_num << endl;
	 cout << "测试集上的准确率是:" << accuracy *100<<"%"<< endl;
	 return accuracy;
}
