#include "icpPointToPlane.h"

#include <Eigen\Eigen>

IcpPointToPlane::IcpPointToPlane(double ** M, int32_t const M_num, int32_t const dim,
	int32_t const neighbour_num, double const flatness)
	:Icp(M, M_num, dim)
{
	computeNormals(neighbour_num, flatness);
}

void IcpPointToPlane::computeNormals(int32_t const neighbour_num, double const flatness)
{
	normals = (Eigen::Vector3d*)malloc(numPts * sizeof(Eigen::Vector3d));
	ANNpoint queryPts = annAllocPt(dim);
	ANNidxArray nnidx = new ANNidx[neighbour_num];
	ANNdistArray dists = new ANNdist[neighbour_num];
	for (size_t i = 0; i < numPts; i++)
	{

		queryPts = dataPts[i];
		kdTree->annkSearch(queryPts, neighbour_num, nnidx, dists);
		normals[i] = computeNormal(nnidx, neighbour_num);
		//std::cout << normals[i] << std::endl;
	}

}

Eigen::Vector3d IcpPointToPlane::computeNormal(ANNidxArray nnidx, int32_t neighbour_num)
{
	double mean_x = 0.0;
	double mean_y = 0.0;
	double mean_z = 0.0;
	for (size_t i = 0; i < neighbour_num; i++)
	{
		mean_x += dataPts[nnidx[i]][0];
		mean_y += dataPts[nnidx[i]][1];
		mean_z += dataPts[nnidx[i]][2];
	}
	mean_x /= (double)neighbour_num;
	mean_y /= (double)neighbour_num;
	mean_z /= (double)neighbour_num;
	Eigen::MatrixXd matA(dim, neighbour_num);
	for (size_t i = 0; i < neighbour_num; i++)
	{
		matA(0, i) = dataPts[nnidx[i]][0] - mean_x;
		matA(1, i) = dataPts[nnidx[i]][1] - mean_y;
		matA(2, i) = dataPts[nnidx[i]][2] - mean_z;
	}
	Eigen::Vector3d normalVector;

#ifdef _SVD
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(matA * matA.transpose(),
		Eigen::ComputeFullU);
	normalVector = svd.matrixU().col(2);

#else
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigenSolver(matA * matA.transpose());
	Eigen::Vector3d eigenValues = eigenSolver.eigenvalues();
	int minIndex = 0;
	if (abs(eigenValues[1]) < abs(eigenValues[minIndex]))
		minIndex = 1;
	if (abs(eigenValues[2]) < abs(eigenValues[minIndex]))
		minIndex = 2;
	normalVector = eigenSolver.eigenvectors().col(minIndex);
	normalVector.normalize();
#endif

	return normalVector;
}

double IcpPointToPlane::fitStep(double ** T, int32_t T_num, Eigen::MatrixXd & R, Eigen::VectorXd & t, std::vector<int> const & active)
{
	int	i;
	int nact = (int)active.size();
	Eigen::MatrixXd p_m(nact, dim);
	Eigen::MatrixXd t_m(nact, dim);	

	if (dim == 2)
	{
		//TODO	
	}
	else
	{
		double r00 = R(0, 0); double r01 = R(0, 1); double r02 = R(0, 2);
		double r10 = R(1, 0); double r11 = R(1, 1); double r12 = R(1, 2);
		double r20 = R(2, 0); double r21 = R(2, 1); double r22 = R(2, 2);
		double t0 = t(0); double t1 = t(1); double t2 = t(2);


		Eigen::MatrixXd A(nact, 6);
		Eigen::VectorXd b(nact);

		ANNpoint queryPt = annAllocPt(dim);
		ANNidxArray nnidx = new ANNidx[1];
		ANNdistArray dists = new ANNdist[1];
		for (size_t i = 0; i < nact; i++)
		{
			int32_t idx = active[i];
			queryPt[0] = r00 * T[idx][0] + r01 * T[idx][1] + r02 * T[idx][2] + t0;
			queryPt[1] = r10 * T[idx][0] + r11 * T[idx][1] + r12 * T[idx][2] + t1;
			queryPt[2] = r20 * T[idx][0] + r21 * T[idx][1] + r22 * T[idx][2] + t2;
			
			kdTree->annkSearch(queryPt, 1, nnidx,dists);

			double dx = dataPts[nnidx[0]][0];
			double dy = dataPts[nnidx[0]][1];
			double dz = dataPts[nnidx[0]][2];

			double nx = normals[nnidx[0]](0);
			double ny = normals[nnidx[0]](1);
			double nz = normals[nnidx[0]](2);
			
			double sx = queryPt[0];
			double sy = queryPt[1];
			double sz = queryPt[2];

			A(i, 0) = nz * sy - ny * sz;
			A(i, 1) = nx * sz - nz * sx;
			A(i, 2) = ny * sx - nx * sy;
			A(i, 3) = nx;
			A(i, 4) = ny;
			A(i, 5) = nz;
			b(i) = nx * dx + ny * dy + nz * dz - nx * sx - ny * sy - nz * sz;

		}

		Eigen::VectorXd x;
#ifdef _SVD
		x = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
		
#else

#endif 

		Eigen::MatrixXd R_ = Eigen::MatrixXd::Identity(3, 3);
		R_(0, 1) = -x(2); R_(1, 0) = x(2);
		R_(0, 2) = x(1); R_(2, 0) = -x(1);
		R_(1, 2) = -x(0); R_(2, 1) = x(0);

		Eigen::JacobiSVD<Eigen::Matrix3d> svd(R_, Eigen::ComputeFullU | Eigen::ComputeFullV);
		R_ = svd.matrixU() * svd.matrixV().transpose();
		
		if (R_.determinant() < 0)
		{
			//TODO
		}
		Eigen::Vector3d t_;
		t_(0) = x(3); t_(1) = x(4); t_(2) = x(5);

		R = R_ * R;
		t = R_ * t + t_;

		return std::max((R_ - Eigen::Matrix3d::Identity()).squaredNorm(), t_.squaredNorm());

	}
	return 0;
}
