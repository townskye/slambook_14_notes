#include <Eigen/Core>
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"

#include "ceres/autodiff.h"

#include "tools/rotation.h"
#include "common/projection.h"

// 相机内参+姿态：参数的维度为9
class VertexCameraBAL : public g2o::BaseVertex<9,Eigen::VectorXd>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexCameraBAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    // 重置估计值 _estimate
    virtual void setToOriginImpl() {}

    virtual void oplusImpl ( const double* update )
    {
      /*
       * VectorXd是动态矩阵，构造时要传入维度
       * VertexCameraBAL::Dimension就是顶点的维度变量（9）
       * 以下语句就是讲数组转化为eigen中的矩阵
      */
      Eigen::VectorXd::ConstMapType v ( update, VertexCameraBAL::Dimension );
        _estimate += v;
    }

};

// 空间点坐标
class VertexPointBAL : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexPointBAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl ( const double* update )
    {
        Eigen::Vector3d::ConstMapType v ( update );
        _estimate += v;
    }
};

class EdgeObservationBAL : public g2o::BaseBinaryEdge<2, Eigen::Vector2d,VertexCameraBAL, VertexPointBAL>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeObservationBAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    // 覆盖基类函数，使用operator()计算代价函数
    virtual void computeError() override   // The virtual function comes from the Edge base class. Must define if you use edge.
    {
        const VertexCameraBAL* cam = static_cast<const VertexCameraBAL*> ( vertex ( 0 ) );
        const VertexPointBAL* point = static_cast<const VertexPointBAL*> ( vertex ( 1 ) );

        // *this：指向当前类的对象（实例）的指针
	// ( *this )：调用当前类的()运算符，即下面重载的()运算符，后面是传入的参数
	( *this ) ( cam->estimate().data(), point->estimate().data(), _error.data() );

    }

    // 重载该类的()运算符
    template<typename T>
    bool operator() ( const T* camera, const T* point, T* residuals ) const
    {
        T predictions[2];
        CamProjectionWithDistortion ( camera, point, predictions );
	// 此处误差定义用预测-测量，与式子10.42相反，与式子10.44相同
        residuals[0] = predictions[0] - T ( measurement() ( 0 ) );
        residuals[1] = predictions[1] - T ( measurement() ( 1 ) );

        return true;
    }


    // 使用Ceres自动求导工具
    virtual void linearizeOplus() override
    {
        // use numeric Jacobians
        // BaseBinaryEdge<2, Vector2d, VertexCameraBAL, VertexPointBAL>::linearizeOplus();
        // return;
        
        // using autodiff from ceres. Otherwise, the system will use g2o numerical diff for Jacobians

        const VertexCameraBAL* cam = static_cast<const VertexCameraBAL*> ( vertex ( 0 ) );
        const VertexPointBAL* point = static_cast<const VertexPointBAL*> ( vertex ( 1 ) );
        typedef ceres::internal::AutoDiff<EdgeObservationBAL, double, VertexCameraBAL::Dimension, VertexPointBAL::Dimension> BalAutoDiff;

        // Dimension：优化边的维度（误差维度2）, RowMajor：表示在内存中按行优先的方式存储
	Eigen::Matrix<double, Dimension, VertexCameraBAL::Dimension, Eigen::RowMajor> dError_dCamera;
        Eigen::Matrix<double, Dimension, VertexPointBAL::Dimension, Eigen::RowMajor> dError_dPoint;
        // 相机参数+空间点坐标=优化变量
	// Matrix.data()：是指向矩阵头部位置的指针
	double *parameters[] = { const_cast<double*> ( cam->estimate().data() ), const_cast<double*> ( point->estimate().data() ) };
        double *jacobians[] = { dError_dCamera.data(), dError_dPoint.data() };
        double value[Dimension];
        bool diffState = BalAutoDiff::Differentiate ( *this, parameters, Dimension, value, jacobians );

        // copy over the Jacobians (convert row-major -> column-major)
        if ( diffState )
        {
            _jacobianOplusXi = dError_dCamera;
            _jacobianOplusXj = dError_dPoint;
        }
        else
        {
            assert ( 0 && "Error while differentiating" );
            _jacobianOplusXi.setZero();
            _jacobianOplusXj.setZero();
        }
    }
};
