#include "icp.h"



Icp::Icp(double ** M, int32_t const M_num, int32_t const dim)
	:dim(dim), numPts(M_num), max_iter(200), min_delta(1e-4)
{
	if (dim != 2 && dim != 3)
	{
		std::invalid_argument("Only support data of dimensionality 2 or 3!");
		kdTree = 0;
		return;
	}

	if (M_num < 5)
	{
		std::invalid_argument("Two few points!");
		kdTree = 0;
		return;
	}
	dataPts = M;
	kdTree = new ANNkd_tree(M, M_num, dim);


}

Icp::~Icp()
{
	if (kdTree)
		delete kdTree;
}

void Icp::fit(double ** T, int32_t T_num, Eigen::MatrixXd& R, Eigen::VectorXd & t, double const indist)
{
	if (!kdTree)
		throw std::invalid_argument("No model available!");

	if (T_num < 5)
		throw std::invalid_argument("Two few template points!");

	std::vector<int> active;
	if (indist <= 0)
	{
		active.clear();
		for (size_t i = 0; i < T_num; i++)
		{
			active.push_back(i);
		}
	}
	else
	{
		//TODO
	}

	fitIterate(T, T_num, R, t, active);

}

void Icp::fitIterate(double ** T, int32_t T_num, Eigen::MatrixXd& R, Eigen::VectorXd & t, std::vector<int>& active)
{
	if (active.size() < 5)
		return;
	for (size_t i = 0; i < max_iter; i++)
	{
		if (fitStep(T, T_num, R, t, active) < min_delta)
			break;
	}
}
