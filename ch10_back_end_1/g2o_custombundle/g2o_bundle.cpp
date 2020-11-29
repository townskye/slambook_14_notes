#include <Eigen/StdVector>
#include <Eigen/Core>

#include <iostream>
#include <stdint.h>

#include <unordered_set>
#include <memory>
#include <vector>
#include <stdlib.h> 

#include "g2o/stuff/sampler.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/batch_stats.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_dogleg.h"

#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/solvers/pcg/linear_solver_pcg.h"
#include "g2o/types/sba/types_six_dof_expmap.h"

#include "g2o/solvers/structure_only/structure_only_solver.h"

#include "common/BundleParams.h"
#include "common/BALProblem.h"
#include "g2o_bal_class.h"


using namespace Eigen;
using namespace std;

// Map类用于通过C++中普通的连续指针或者数组来构造Eigen里的Matrix类
// 构造时需要传入指针和数据的长度
typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;
typedef g2o::BlockSolver<g2o::BlockSolverTraits<9,3> > BalBlockSolver;

// set up the vertexs and edges for the bundle adjustment. 
void BuildProblem(const BALProblem* bal_problem, g2o::SparseOptimizer* optimizer, const BundleParams& params)
{
    const int num_points = bal_problem->num_points();
    const int num_cameras = bal_problem->num_cameras();
    const int camera_block_size = bal_problem->camera_block_size();  // 相机参数维度
    const int point_block_size = bal_problem->point_block_size();  // 空间点维度

    // Set camera vertex with initial value in the dataset.
    const double* raw_cameras = bal_problem->cameras();
    for(int i = 0; i < num_cameras; ++i)
    {
        // raw_cameras是一个指针，下一语句就是利用该指针构造Vector类型数据
        ConstVectorRef temVecCamera(raw_cameras + camera_block_size * i,camera_block_size);
        VertexCameraBAL* pCamera = new VertexCameraBAL();
        pCamera->setEstimate(temVecCamera);   // initial value for the camera i..
        pCamera->setId(i);                    // set id for each camera vertex 
  
        // remeber to add vertex into optimizer..
        optimizer->addVertex(pCamera);
        
    }

    // Set point vertex with initial value in the dataset. 
    const double* raw_points = bal_problem->points();
    // const int point_block_size = bal_problem->point_block_size();
    for(int j = 0; j < num_points; ++j)
    {
        ConstVectorRef temVecPoint(raw_points + point_block_size * j, point_block_size);
        VertexPointBAL* pPoint = new VertexPointBAL();
        pPoint->setEstimate(temVecPoint);   // initial value for the point i.. 点云的初始值
        pPoint->setId(j + num_cameras);     // each vertex should have an unique id, no matter it is a camera vertex, or a point vertex 

        // remeber to add vertex into optimizer..
        pPoint->setMarginalized(true);  // 设置该点在解方程时进行Schur消元
        optimizer->addVertex(pPoint);
    }

    // Set edges for graph..
    // 将一个相机与一个被观测点的组合视为一次观测
    const int  num_observations = bal_problem->num_observations();
    const double* observations = bal_problem->observations();   // pointer for the first observation..

    for(int i = 0; i < num_observations; ++i)
    {
        EdgeObservationBAL* bal_edge = new EdgeObservationBAL();

        const int camera_id = bal_problem->camera_index()[i]; // get id for the camera; 
        const int point_id = bal_problem->point_index()[i] + num_cameras; // get id for the point 
        
        // 鲁棒核(Huber loss)函数（默认为不用设置）
        if(params.robustify)
        {
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            rk->setDelta(1.0);
            bal_edge->setRobustKernel(rk);
        }
        // set the vertex by the ids for an edge observation
        bal_edge->setVertex(0,dynamic_cast<VertexCameraBAL*>(optimizer->vertex(camera_id)));
        bal_edge->setVertex(1,dynamic_cast<VertexPointBAL*>(optimizer->vertex(point_id)));
	// 设置协方差矩阵（在此处为单位矩阵）
        bal_edge->setInformation(Eigen::Matrix2d::Identity());
        bal_edge->setMeasurement(Eigen::Vector2d(observations[2*i+0],observations[2*i + 1]));

       optimizer->addEdge(bal_edge) ;
    }

}

void WriteToBALProblem(BALProblem* bal_problem, g2o::SparseOptimizer* optimizer)
{
    const int num_points = bal_problem->num_points();
    const int num_cameras = bal_problem->num_cameras();
    const int camera_block_size = bal_problem->camera_block_size();
    const int point_block_size = bal_problem->point_block_size();

    double* raw_cameras = bal_problem->mutable_cameras();
    for(int i = 0; i < num_cameras; ++i)
    {
        VertexCameraBAL* pCamera = dynamic_cast<VertexCameraBAL*>(optimizer->vertex(i));
        Eigen::VectorXd NewCameraVec = pCamera->estimate();
	// memcpy(destination, source, size)：内存拷贝函数
	// 从源地址所指的内存位置开始拷贝size个数据到目的地址所指的内存位置
	// 将优化器中当前优化变量的值写入BAL的数据存储中（覆盖原始raw参数）
        memcpy(raw_cameras + i * camera_block_size, NewCameraVec.data(), sizeof(double) * camera_block_size);
    }

    double* raw_points = bal_problem->mutable_points();
    for(int j = 0; j < num_points; ++j)
    {
        VertexPointBAL* pPoint = dynamic_cast<VertexPointBAL*>(optimizer->vertex(j + num_cameras));
        Eigen::Vector3d NewPointVec = pPoint->estimate();
        memcpy(raw_points + j * point_block_size, NewPointVec.data(), sizeof(double) * point_block_size);
    }
}

//this function is  unused yet..
// void SetMinimizerOptions(std::shared_ptr<BalBlockSolver>& solver_ptr, const BundleParams& params, g2o::SparseOptimizer* optimizer)
// {
//     //std::cout<<"Set Minimizer  .."<< std::endl;
//     g2o::OptimizationAlgorithmWithHessian* solver;
//     if(params.trust_region_strategy == "levenberg_marquardt"){
//         solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr.get());
//     }
//     else if(params.trust_region_strategy == "dogleg"){
//         solver = new g2o::OptimizationAlgorithmDogleg(solver_ptr.get());
//     }
//     else 
//     {
//         std::cout << "Please check your trust_region_strategy parameter again.."<< std::endl;
//         exit(EXIT_FAILURE);
//     }
// 
//     optimizer->setAlgorithm(solver);
//     //std::cout<<"Set Minimizer  .."<< std::endl;
// }

//this function is  unused yet..
// void SetLinearSolver(std::shared_ptr<BalBlockSolver>& solver_ptr, const BundleParams& params)
// {
//     //std::cout<<"Set Linear Solver .."<< std::endl;
//     g2o::LinearSolver<BalBlockSolver::PoseMatrixType>* linearSolver = 0;
//     
//     if(params.linear_solver == "dense_schur" ){
//         linearSolver = new g2o::LinearSolverDense<BalBlockSolver::PoseMatrixType>();
//     }
//     else if(params.linear_solver == "sparse_schur"){
//         linearSolver = new g2o::LinearSolverCholmod<BalBlockSolver::PoseMatrixType>();
//         dynamic_cast<g2o::LinearSolverCholmod<BalBlockSolver::PoseMatrixType>* >(linearSolver)->setBlockOrdering(true);  // AMD ordering , only needed for sparse cholesky solver
//     }
//     
// 
//     solver_ptr = std::make_shared<BalBlockSolver>(linearSolver);
//     std::cout <<  "Set Complete.."<< std::endl;
// }

void SetSolverOptionsFromFlags(BALProblem* bal_problem, const BundleParams& params, g2o::SparseOptimizer* optimizer)
{   
    // BalBlockSolver* solver_ptr;
    
    // g2o::LinearSolver<BalBlockSolver::PoseMatrixType>* linearSolver = 0;
    std::unique_ptr< g2o::LinearSolver<BalBlockSolver::PoseMatrixType> > linearSolver;
    
    if(params.linear_solver == "dense_schur" ){
        // linearSolver = new g2o::LinearSolverDense<BalBlockSolver::PoseMatrixType>();

        // std::unique_ptr< g2o::LinearSolver<BalBlockSolver::PoseMatrixType> > linearSolver_copy ( new g2o::LinearSolverDense<BalBlockSolver::PoseMatrixType>() );
        // linearSolver = std::move( linearSolver_copy );

        linearSolver.reset( new g2o::LinearSolverDense<BalBlockSolver::PoseMatrixType>() );
    }
    else if(params.linear_solver == "sparse_schur"){
        // linearSolver = new g2o::LinearSolverCholmod<BalBlockSolver::PoseMatrixType>();
        // dynamic_cast< g2o::LinearSolverCholmod<BalBlockSolver::PoseMatrixType>* >(linearSolver)->setBlockOrdering(true);  // AMD ordering , only needed for sparse cholesky solver
        // 让solver对矩阵进行排序保持稀疏性
        auto linearSolver_old = g2o::make_unique<g2o::LinearSolverCholmod<BalBlockSolver::PoseMatrixType>>();
        linearSolver_old->setBlockOrdering(true);
        linearSolver = std::move(linearSolver_old);
    }
    
    std::unique_ptr<BalBlockSolver> solver_ptr ( new BalBlockSolver( std::move(linearSolver) ) );
    //SetLinearSolver(solver_ptr, params);

    //SetMinimizerOptions(solver_ptr, params, optimizer);
    g2o::OptimizationAlgorithmWithHessian* solver;
    if(params.trust_region_strategy == "levenberg_marquardt"){
        // solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        solver = new g2o::OptimizationAlgorithmLevenberg( std::move(solver_ptr) );
    }
    else if(params.trust_region_strategy == "dogleg"){
        // solver = new g2o::OptimizationAlgorithmDogleg(solver_ptr);
        solver = new g2o::OptimizationAlgorithmDogleg( std::move(solver_ptr) );
    }
    else 
    {
        std::cout << "Please check your trust_region_strategy parameter again.."<< std::endl;
        exit(EXIT_FAILURE);
    }

    optimizer->setAlgorithm(solver);
}

// SolveProblem的形参filename是原始的数据文件名
void SolveProblem(const char* filename, const BundleParams& params)
{
    BALProblem bal_problem(filename);

    // show some information here ...
    std::cout << "bal problem file loaded..." << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
              << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observatoins. " << std::endl;

    // store the initial 3D cloud points and camera pose..
    if(!params.initial_ply.empty()){
        bal_problem.WriteToPLYFile(params.initial_ply);
    }

    std::cout << "beginning problem..." << std::endl;
    
    // add some noise for the intial value
    srand(params.random_seed);
    bal_problem.Normalize();
    bal_problem.Perturb(params.rotation_sigma, params.translation_sigma,
                        params.point_sigma);

    std::cout << "Normalization complete..." << std::endl;


    g2o::SparseOptimizer optimizer;
    SetSolverOptionsFromFlags(&bal_problem, params, &optimizer);
    BuildProblem(&bal_problem, &optimizer, params);

    
    std::cout << "begin optimizaiton .."<< std::endl;
    // perform the optimizaiton 
    optimizer.initializeOptimization();
    optimizer.setVerbose(true);
    optimizer.optimize(params.num_iterations);

    std::cout << "optimization complete.. "<< std::endl;
    // write the optimized data into BALProblem class
    WriteToBALProblem(&bal_problem, &optimizer);

    // write the result into a .ply file.
    if(!params.final_ply.empty()){
        bal_problem.WriteToPLYFile(params.final_ply);
    }
   
}

int main(int argc, char** argv)
{
    
    BundleParams params(argc,argv);  // set the parameters here.

    if(params.input.empty()){
        std::cout << "Usage: bundle_adjuster -input <path for dataset>";
        return 1;
    }

    SolveProblem(params.input.c_str(), params);
  
    return 0;
}
