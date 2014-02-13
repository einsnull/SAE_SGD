#pragma once
#include <Eigen/Dense>
#include <ctime>
#include <fstream>

#define GD 1
#define MINI_BATCH_SGD 0

using namespace std;
using namespace Eigen;

//scalar reciprocal
double reciprocalScalar(double x)
{
	return 1.0/x;
}

//scalar sigmoid function
double sigmoidScalar(double x)
{
	return 1.0 / (1 + exp(-x));
}

//scalar log function
double logScalar(double x)
{
	return log(x);
}

// sparse auto encoder class
class SAE
{
private:
	//weights
	MatrixXd theta1;
	MatrixXd theta2;
	//bias
	MatrixXd b1;
	MatrixXd b2;
	int inputSize;
	int hiddenSize;
public:
	SAE(int inputSize,int hiddenSize);
	void train(
		MatrixXd &trainData,double lambda,
		double alpha,double beta,double sp,int maxIter,
		int trainMode,void *option);
	bool saveModel(char *szFileName);
	bool loadModel(char *szFileName);
	MatrixXd getTheta();
	MatrixXd getBias();
	
private:
	MatrixXd reciprocal(MatrixXd &z);
	MatrixXd randomInitialize(int lIn,int lOut);
	inline MatrixXd sigmoid(MatrixXd &z);
	MatrixXd SAE::sigmoidGradient(MatrixXd &z);
	inline MatrixXd logMat(MatrixXd &z);
	void updateParameters(
		MatrixXd &theta1Grad1,MatrixXd &theta2Grad2,
		MatrixXd &b1Grad,MatrixXd &b2Grad,double alpha);
	void gradientDescent(MatrixXd &trainData,
		double lambda,double alpha,int maxIter,double beta,double sp);
	void miniBatchSGD(MatrixXd &trainData,double lambda,
		double alpha,int maxIter,int batchSize,double beta,double sp);
	double computeCost(MatrixXd &data,double lambda,MatrixXd &theta1Grad,
		MatrixXd &theta2Grad,MatrixXd &b1Grad,MatrixXd &b2Grad,
		double beta,double sparsityParam);
};

//constructor
SAE::SAE(int inputSize,int hiddenSize)
{
	this->inputSize = inputSize;
	this->hiddenSize = hiddenSize;
	theta1 = randomInitialize(hiddenSize,inputSize);
	theta2 = randomInitialize(inputSize,hiddenSize);
	b1 = MatrixXd::Zero(hiddenSize,1);
	b2 = MatrixXd::Zero(inputSize,1);
}

MatrixXd SAE::getTheta()
{
	return theta1;
}

MatrixXd SAE::getBias()
{
	return b1;
}

//return 1.0 ./ z
MatrixXd SAE::reciprocal(MatrixXd &z)
{
	return z.unaryExpr(ptr_fun(reciprocalScalar));
}

//random initialize the weights
MatrixXd SAE::randomInitialize(int lIn,int lOut)
{
	//random initialize the weight
	int i,j;
	double epsilon = sqrt(6.0/(this->inputSize + this->hiddenSize + 1));
	MatrixXd result(lIn,lOut);
	srand((unsigned int)time(NULL));
	for(i = 0;i < lOut;i++)
	{
		for(j = 0;j < lIn;j++)
		{
			result(j,i) = ((double)rand() / (double)RAND_MAX) * 2 * epsilon - epsilon; 
		}
	}
	return result;
}

//component wise sigmoid function
MatrixXd SAE::sigmoid(MatrixXd &z)
{
	return z.unaryExpr(ptr_fun(sigmoidScalar));
}

MatrixXd SAE::sigmoidGradient(MatrixXd &z)
{
	//return sigmoid(z) .* (1 - sigmoid(z))
	MatrixXd result;
	MatrixXd sigm = sigmoid(z);
	MatrixXd item = MatrixXd::Ones(z.rows(),z.cols()) - sigm;
	result = sigm.cwiseProduct(item);
	return result;
}

//component wise log function
MatrixXd SAE::logMat(MatrixXd &z)
{
	return z.unaryExpr(ptr_fun(logScalar));
}

//gradient descent update rule
void SAE::updateParameters(
	MatrixXd &theta1Grad1,MatrixXd &theta2Grad2,
	MatrixXd &b1Grad,MatrixXd &b2Grad,double alpha)
{
	theta1 -= theta1Grad1*alpha;
	theta2 -= theta2Grad2*alpha;
	b1 -= b1Grad * alpha;
	b2 -= b2Grad * alpha;
}

//cost function
double SAE::computeCost(
	MatrixXd &data,double lambda,MatrixXd &theta1Grad,MatrixXd &theta2Grad,
	MatrixXd &b1Grad,MatrixXd &b2Grad,double beta,double sparsityParam)
{

	double cost = 0;

	int numOfExamples = data.cols();

	MatrixXd a1 = data;
	
	MatrixXd z2 = theta1 * data + b1.replicate(1,numOfExamples);
	MatrixXd a2 = sigmoid(z2);
	MatrixXd z3 = theta2 * a2 + b2.replicate(1,numOfExamples);
	MatrixXd a3 = sigmoid(z3);
	
	MatrixXd rho = a2.rowwise().sum() * (1.0 / (double)numOfExamples);
	double sp = sparsityParam;
	MatrixXd term1 = MatrixXd::Ones(rho.rows(),rho.cols()) - rho;
	MatrixXd spDelta = reciprocal(term1) * (1.0 - sp)
		-  reciprocal(rho) * sp;
	//compute delta
	MatrixXd delta3 = (a3 - data).cwiseProduct(sigmoidGradient(z3));

	MatrixXd delta2 = (theta2.transpose() * delta3 
		+ spDelta.replicate(1,numOfExamples)).cwiseProduct(sigmoidGradient(z2)) * beta;

	//compute gradients

	theta2Grad = delta3 * a2.transpose() * (1.0 / (double)numOfExamples)
		+ theta2 * lambda;
	b2Grad = delta3.rowwise().sum() * (1.0 / (double)numOfExamples);
	theta1Grad = delta2 * a1.transpose() * ( 1.0 / (double)numOfExamples)
		+ theta1 * lambda;
	b1Grad = delta2.rowwise().sum() * (1.0  / (double)numOfExamples);

	MatrixXd term2 = reciprocal(rho) * sp;
	MatrixXd term3 = reciprocal(term1) * (1.0 - sp);
	//compute cost
	double spCost = (logMat(term2) * sp + logMat(term3) * (1.0 - sp)).array().sum() * beta;
	
	double regCost = (theta1.array().square().sum()
		+ theta2.array().square().sum()) * (lambda / 2.0);
	
	cost = (a3 - data).array().square().sum() * (1.0 / 2.0 / numOfExamples)
		+ regCost + spCost;
	return cost;
}


//gradient descent method
void SAE::gradientDescent(
	MatrixXd &trainData,double lambda,double alpha,int maxIter,double beta,double sp)
{
	double J = 0;
	MatrixXd theta1Grad(theta1.rows(),theta1.cols());
	MatrixXd theta2Grad(theta2.rows(),theta2.cols());
	MatrixXd b1Grad(b1.rows(),b1.cols());
	MatrixXd b2Grad(b2.rows(),b2.cols());
	int iter = 1;
	//gradient decent
	for(int i = 0; i < maxIter;i++)
	{
		// compute the cost
		J = computeCost(trainData,lambda,theta1Grad,theta2Grad,b1Grad,b2Grad,beta,sp);
#ifdef _IOSTREAM_
		cout << "iter: " << iter++ << "  cost: " << J << endl;
#endif
		if(fabs(J) < 0.001)
		{
			break;
		}
		updateParameters(theta1Grad,theta2Grad,b1Grad,b2Grad,alpha);
	}
}

//mini batch stochastic gradient descent
void SAE::miniBatchSGD(
	MatrixXd &trainData,double lambda,double alpha,int maxIter,int batchSize,double beta,double sp)
{
	//get the binary code of labels
	MatrixXd theta1Grad(theta1.rows(),theta1.cols());
	MatrixXd theta2Grad(theta2.rows(),theta2.cols());
	MatrixXd b1Grad(b1.rows(),b1.cols());
	MatrixXd b2Grad(b2.rows(),b2.cols());
	MatrixXd miniTrainData(trainData.rows(),batchSize);
	int iter = 1;
	int numBatches = trainData.cols() / batchSize;
	
	//mini batch stochastic gradient decent
	for(int i = 0; i < maxIter;i++)
	{
		double J = 0;
		// compute the cost
		for(int j = 0;j < numBatches; j++)
		{
			miniTrainData = trainData.middleCols(j * batchSize,batchSize);
			J += computeCost(miniTrainData,lambda,theta1Grad,theta2Grad,
				b1Grad,b2Grad,beta,sp);
#ifdef _IOSTREAM_
			if(miniTrainData.cols() < 1 || miniTrainData.rows() < 1)
			{
				cout << "Too few training examples!"  << endl; 
			}
#endif

			if(fabs(J) < 0.001)
			{
				break;
			}
			updateParameters(theta1Grad,theta2Grad,b1Grad,b2Grad,alpha);
		}
		J = J / numBatches;
#ifdef _IOSTREAM_
		cout << "iter: " << iter++ << "  cost: " << J << endl;
#endif
	}
}

//train the model
void SAE::train(
	MatrixXd &trainData,double lambda,
	double alpha,double beta,double sp,int maxIter,
	int trainMode,void *option)
{
	if(trainData.rows() != this->inputSize)
	{
#ifdef _IOSTREAM_
		cout << "TrainData rows:" << trainData.rows() << endl;
		cout << "dimension mismatch!" << endl;
#endif
		return;
	}
	if(trainMode == GD)
	{
		gradientDescent(trainData,lambda,alpha,maxIter,beta,sp);
	}
	else if(trainMode == MINI_BATCH_SGD)
	{
		miniBatchSGD(trainData,lambda,alpha,maxIter,*((int*)option),beta,sp);
	}
}


//save model to file
bool SAE::saveModel(char *szFileName)
{
	ofstream ofs(szFileName);
	if(!ofs)
	{
		return false;
	}
	int i,j;
	ofs << inputSize << " " << hiddenSize << endl;
	for(i = 0; i < theta1.rows(); i++)
	{
		for(j = 0;j < theta1.cols(); j++)
		{
			ofs << theta1(i,j) << " ";
		}
	}
	ofs << endl;
	for(i = 0; i < theta2.rows(); i++)
	{
		for(j = 0;j < theta2.cols(); j++)
		{
			ofs << theta2(i,j) << " ";
		}
	}
	ofs << endl;
	for(i = 0; i < b1.rows(); i++)
	{
		for(j = 0; j < b1.cols(); j++) 
		{
			ofs << b1(i,j) << " ";
		}
	}
	ofs << endl;
	for(i = 0; i < b2.rows(); i++)
	{
		for(j = 0; j < b2.cols(); j++) 
		{
			ofs << b2(i,j) << " ";
		}
	}
	ofs.close();
	return true;
}

//load model from file
bool SAE::loadModel(char *szFileName)
{
	ifstream ifs(szFileName);
	if(!ifs)
	{
		return false;
	}
	ifs >> this -> inputSize >> this -> hiddenSize;
	int i,j;
	theta1.resize(this->hiddenSize,this->inputSize);
	theta2.resize(this->inputSize,this->hiddenSize);
	b1.resize(1,hiddenSize);
	b2.resize(1,inputSize);

	for(i = 0; i < theta1.rows(); i++)
	{
		for(j = 0;j < theta1.cols(); j++)
		{
			if(ifs.eof())
			{
				ifs.close();
				return false;
			}
			ifs >> theta1(i,j);
		}
	}
	for(i = 0; i < theta2.rows(); i++)
	{
		for(j = 0;j < theta2.cols(); j++)
		{
			if(ifs.eof())
			{
				ifs.close();
				return false;
			}
			ifs >> theta2(i,j);
		}
	}
	for(i = 0; i < b1.rows(); i++)
	{
		for(j = 0; j < b1.cols(); j++) 
		{
			if(ifs.eof())
			{
				ifs.close();
				return false;
			}
			ifs >> b1(i,j);
		}
	}
	for(i = 0; i < b2.rows(); i++)
	{
		for(j = 0; j < b2.cols(); j++) 
		{
			if(ifs.eof())
			{
				ifs.close();
				return false;
			}
			ifs >> b2(i,j);
		}
	}
	ifs.close();
	return true;
}