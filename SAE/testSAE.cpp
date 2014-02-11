#include <iostream>
#include "SAE.h"
#include <opencv2/opencv.hpp>
#include "cmath"
#include "getConfig.h"

using namespace std;

#pragma comment(lib,"opencv_core243.lib")
#pragma comment(lib,"opencv_highgui243.lib")
#pragma comment(lib,"opencv_imgproc243.lib")

bool loadDataSet(MatrixXd &data);

void buildImage(MatrixXd theta,int imgWidth,char *szFileName,bool showFlag = false);

int main()
{
	//regularization coefficient
	double lambda = 0.0001;
	//learning rate
	double alpha = 0.03;
	double beta = 3;
	double sp = 0.01;
	int maxIter = 15000;
	int miniBatchSize = 1000;
	int n = 2;
	int inputSize = 64;
	int hiddenSize = 33;
	int imgWidth = 8; 
	char *fileBuf  = new char[4096];
	bool ret = loadFileToBuf("ParamConfig.ini",fileBuf,4096);
	if(ret)
	{
		getConfigDoubleValue(fileBuf,"lambda:",lambda);
		getConfigDoubleValue(fileBuf,"alpha:",alpha);
		getConfigDoubleValue(fileBuf,"beta:",beta);
		getConfigDoubleValue(fileBuf,"sparseParam:",sp);
		getConfigIntValue(fileBuf,"maxIter:",maxIter);
		getConfigIntValue(fileBuf,"miniBatchSize:",miniBatchSize);
		getConfigIntValue(fileBuf,"hiddenSize:",hiddenSize);
		getConfigIntValue(fileBuf,"inputSize:",inputSize);
		getConfigIntValue(fileBuf,"imgWidth:",imgWidth);
		cout << "lambda:" << lambda << endl;
		cout << "alpha:" << alpha << endl;
		cout << "beta:" << beta << endl;
		cout << "sparseParam:" << sp << endl;
		cout << "maxIter:" << maxIter << endl;
		cout << "miniBatchSize:" << miniBatchSize << endl;
		cout << "hiddenSize:" << hiddenSize << endl;
		cout << "inputSize:" << inputSize << endl;
		cout << "imgWidth:" << imgWidth << endl;
	}
	delete []fileBuf;
	MatrixXd tmpData(n,inputSize);
	SAE sae(inputSize,hiddenSize);
	loadDataSet(tmpData);
	//buildImage(trainData.topRows(100),8,"data.bmp");
	//sae.loadModel("SAE_Model.txt");
	clock_t start = clock();
	MatrixXd trainData = tmpData.transpose();
	buildImage(trainData.leftCols(100).transpose(),imgWidth,"data.bmp");
	sae.train(trainData,lambda,alpha,beta,sp,maxIter,MINI_BATCH_SGD,&miniBatchSize);
	buildImage(sae.getTheta(),imgWidth,"weights.bmp");
	clock_t end = clock();
	cout << "The code ran for " << (end - start)/(double)(CLOCKS_PER_SEC*60) << " minutes." << endl;
	sae.saveModel("SAE_Model.txt");
	/*sae.loadModel("SAE_Model.txt");
	buildImage(sae.theta1,imgWidth,"weights.bmp",true);*/
	cout << "lambda:" << lambda << endl;
	cout << "alpha:" << alpha << endl;
	cout << "beta:" << beta << endl;
	cout << "sparseParam:" << sp << endl;
	cout << "miniBatchSize:" << miniBatchSize << endl;
	system("pause");
	return 0;
}


bool loadDataSet(MatrixXd &data)
{
	ifstream ifs("mydata.txt");
	if(!ifs)
	{
		return false;
	}
	cout << "Loading data..." << endl;
	int inputLayerSize;
	ifs >> inputLayerSize;
	int dataSetSize;
	ifs >> dataSetSize;
	data.resize(dataSetSize,inputLayerSize);
	for(int i = 0; i < dataSetSize; i++)
	{
		for(int j = 0; j < inputLayerSize; j++)
		{
			ifs >> data(i,j);
		}
	}
	//data = data - MatrixXd::Ones(data.rows(),data.cols())*0.5;
	ifs.close();
	return true;
}

void buildImage(MatrixXd theta,int imgWidth,char* szFileName,bool showFlag)
{
	int ratio = 10;
	int margin = 1;
	int rows = theta.rows();
	int cols = theta.cols();
	int perRow = (int)(sqrt(rows) + 0.5);
	int tCols = (int)(rows / (double)perRow + 0.5);
	/*cout << "perRow: " << perRow << endl;
	cout << "tCols: " << tCols << endl;*/
	MatrixXd max = theta.rowwise().maxCoeff();
	MatrixXd min = theta.rowwise().minCoeff();

	int imgHeight = cols/imgWidth;
	IplImage* iplImage = cvCreateImage(
		cvSize(imgWidth * perRow + margin * (perRow+1),imgHeight * tCols + margin * (tCols + 1)),
		IPL_DEPTH_8U,1);
	int step = iplImage->widthStep;
	uchar *data = (uchar *)iplImage->imageData;

	for(int i = 0;i < rows;i++)
	{
		int n = 0;
		for(int j = 0;j < imgHeight;j++)
		{
			for(int k = 0;k < imgWidth; k++)
			{
				double per = (theta(i,n) - min(i,0) ) / (max(i,0) - min(i,0));
			
				data[(j + margin + (i / perRow * (imgHeight + margin))) * step + margin + k + (i % perRow)*imgWidth + (i % perRow) * margin] = 
					(uchar)(int)(per * 230.0);
				n ++;
			}
		}
		
	}
	cvSaveImage(szFileName,iplImage);
	if(showFlag)
	{
		cvNamedWindow("Image",CV_WINDOW_AUTOSIZE);
		IplImage* iplImageShow = cvCreateImage(
			cvSize(iplImage->width * ratio,iplImage->height * ratio),IPL_DEPTH_8U,1);
		cvResize(iplImage,iplImageShow,CV_INTER_CUBIC);
		cvShowImage("Image",iplImageShow);
		cvWaitKey(100000);
		cvDestroyWindow("Image");
		cvReleaseImage(&iplImageShow);
		cvReleaseImage(&iplImage);
	}
}


//void buildImage(MatrixXd theta,int imgWidth)
//{
//	int margin = 3;
//	int rows = theta.rows();
//	int cols = theta.cols();
//	MatrixXd max = theta.rowwise().maxCoeff();
//	MatrixXd min = theta.rowwise().minCoeff();
//
//	int imgHeight = cols/imgWidth;
//	IplImage* iplImage = cvCreateImage(cvSize(imgWidth,imgHeight),IPL_DEPTH_8U,1);
//	int step = iplImage->widthStep;
//	uchar *data = (uchar *)iplImage->imageData;
//	
//	for(int i = 0;i < rows;i++)
//	{
//		int n = 0;
//		for(int j = 0;j < imgHeight;j++)
//		{
//			for(int k = 0;k < imgWidth; k++)
//			{
//				double per = (theta(i,n) - min(i,0)) / (max(i,0) - min(i,0));
//				data[j * step + k] = (uchar)(int)(per * 255.0);
//				n ++;
//			}
//		}
//		char szFileName[200];
//		sprintf_s(szFileName,"test%d.bmp",i);
//		cvSaveImage(szFileName,iplImage);
//	}
//	cvReleaseImage(&iplImage);
//}