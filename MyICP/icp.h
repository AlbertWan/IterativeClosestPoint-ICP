#pragma once
#include <iostream>

#include <ANN\ANN.h>

#include <Eigen\Eigen>

class Icp
{
public:
	Icp(double** M, int32_t const M_num, int32_t const dim);

	virtual ~Icp();

	void fit(double** T, int32_t T_num, Eigen::MatrixXd& R, Eigen::VectorXd& t, double const indist);

private:
	void fitIterate(double** T, int32_t T_num, Eigen::MatrixXd& R, Eigen::VectorXd& t, std::vector<int>& active);

	virtual double fitStep(double** T, int32_t T_num, Eigen::MatrixXd& R, Eigen::VectorXd& t, std::vector<int> const& active) = 0;

protected:
	ANNpointArray dataPts;
	ANNkd_tree* kdTree;

	int32_t dim;
	int32_t numPts;
	int32_t max_iter;
	double min_delta;
};

