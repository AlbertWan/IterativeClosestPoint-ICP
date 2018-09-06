#pragma once

#include <Eigen\Eigen>


#include "icp.h"

class IcpPointToPlane :public Icp
{
public:
	IcpPointToPlane(double** M, int32_t const M_num, int32_t const dim,
		int32_t const neighbour_num = 10, double const flatness = 5.0);


private:
	Eigen::Vector3d* normals;

	void computeNormals(int32_t const neighbour_num, double const flatness);

	Eigen::Vector3d computeNormal(ANNidxArray nnidx, int32_t neighbour_num);

	double fitStep(double** T, int32_t T_num, Eigen::MatrixXd& R, Eigen::VectorXd& t, std::vector<int> const& active);
};