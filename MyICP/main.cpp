
#include <iostream>
#include <fstream>
#include <string>

#include <ANN\ANN.h>

#include "icpPointToPlane.h"

int CountLines(char* fileName)
{
	std::ifstream fin(fileName, std::ifstream::in);
	std::string tmp;
	int count = 0;
	while (std::getline(fin, tmp, '\n'))
	{
		count++;
	}
	std::string();
	return count;
}

void GetPoints(char* fileName, ANNpointArray dataPts, int nPts, int dim)
{
	std::ifstream fin(fileName, std::ifstream::in);
	int n = 0;
	while (nPts--)
	{
		fin >> dataPts[n][0];
		fin >> dataPts[n][1];
		fin >> dataPts[n][2];
		n++;
	}
}

int main(int argc,char** argv)
{
	int dim = 3;
	int pts = CountLines(argv[1]);
	std::cout << pts << std::endl;
	ANNpointArray dataPts = annAllocPts(pts, dim);
	GetPoints(argv[1], dataPts, pts, dim);
	IcpPointToPlane i(dataPts, pts, dim);

	Eigen::MatrixXd R = Eigen::Matrix<double, 3, 3>::Identity();
	Eigen::VectorXd t = Eigen::Vector3d::Zero();

	ANNpointArray sourcePts = annAllocPts(pts, dim);
	for (size_t i = 0; i < pts; i++)
	{
		sourcePts[i][0] = dataPts[i][0] + 2;
		sourcePts[i][1] = dataPts[i][1] + 3;
		sourcePts[i][2] = dataPts[i][2] + 4;
	}

	i.fit(sourcePts, pts, R, t, -1);
	std::cout << R << std::endl;
	std::cout << t << std::endl;
}