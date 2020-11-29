#include <iostream>
#include <fstream>
#include "ceres/ceres.h"

#include "SnavelyReprojectionError.h"
#include "common/BALProblem.h"
#include "common/BundleParams.h"


using namespace ceres;

// 选取 linear solver
void SetLinearSolver(ceres::Solver::Options* options, const BundleParams& params)
{
    CHECK(ceres::StringToLinearSolverType(params.linear_solver, &options->linear_solver_type));
    CHECK(ceres::StringToSparseLinearAlgebraLibraryType(params.sparse_linear_algebra_library, &options->sparse_linear_algebra_library_type));
    CHECK(ceres::StringToDenseLinearAlgebraLibraryType(params.dense_linear_algebra_library, &options->dense_linear_algebra_library_type));
    options->num_linear_solver_threads = params.num_threads;

}

// 采用额外的类型ParameterBlockOrdering来设置消元顺序，用AddElementToGroup对变量进行编号，优先消元编号最小的变量
void SetOrdering(BALProblem* bal_problem, ceres::Solver::Options* options, const BundleParams& params)
{
    const int num_points = bal_problem->num_points();
    const int point_block_size = bal_problem->point_block_size();
    double* points = bal_problem->mutable_points();   // 存储点云数据的指针

    const int num_cameras = bal_problem->num_cameras();
    const int camera_block_size = bal_problem->camera_block_size();
    double* cameras = bal_problem->mutable_cameras();


    if (params.ordering == "automatic")
        return;

    ceres::ParameterBlockOrdering* ordering = new ceres::ParameterBlockOrdering;

    // The points come before the cameras
    for(int i = 0; i < num_points; ++i)
       ordering->AddElementToGroup(points + point_block_size * i, 0);
       
    
    for(int i = 0; i < num_cameras; ++i)
        ordering->AddElementToGroup(cameras + camera_block_size * i, 1);

    options->linear_solver_ordering.reset(ordering);

}

// 设置优化相关参数
void SetMinimizerOptions(Solver::Options* options, const BundleParams& params){
    options->max_num_iterations = params.num_iterations;
    options->minimizer_progress_to_stdout = true;
    options->num_threads = params.num_threads;   // 用于计算的线程数目，可以加速雅可比矩阵的计算
    // options->eta = params.eta;
    // options->max_solver_time_in_seconds = params.max_solver_time;
    
    // 下降策略的选取
    CHECK(StringToTrustRegionStrategyType(params.trust_region_strategy,
                                        &options->trust_region_strategy_type));
}

// 给Solver::Options的类型成员变量进行赋值来调整不同的下降策略
void SetSolverOptionsFromFlags(BALProblem* bal_problem,
                               const BundleParams& params, Solver::Options* options){
    SetMinimizerOptions(options,params);
    SetLinearSolver(options,params);
    SetOrdering(bal_problem, options,params);
}

// Problem：构建最小二乘问题
void BuildProblem(BALProblem* bal_problem, Problem* problem, const BundleParams& params)
{
    const int point_block_size = bal_problem->point_block_size();   // 3维
    const int camera_block_size = bal_problem->camera_block_size();   // 9维
    double* points = bal_problem->mutable_points();   // 空间点数据指针
    double* cameras = bal_problem->mutable_cameras();   // 相机姿态数据指针

    // Observations is 2 * num_observations long array observations
    // [u_1, u_2, ... u_n], where each u_i is two dimensional, the x 
    // and y position of the observation. 
    const double* observations = bal_problem->observations();

    for(int i = 0; i < bal_problem->num_observations(); ++i){
        CostFunction* cost_function;

        // Each Residual block takes a point and a camera as input 
        // and outputs a 2 dimensional Residual
      
        cost_function = SnavelyReprojectionError::Create(observations[2*i + 0], observations[2*i + 1]);

        // If enabled use Huber's loss function. 
        LossFunction* loss_function = params.robustify ? new HuberLoss(1.0) : NULL;

        // Each observatoin corresponds to a pair of a camera and a point 
        // which are identified by camera_index()[i] and point_index()[i]
        // respectively.
        double* camera = cameras + camera_block_size * bal_problem->camera_index()[i];
        double* point = points + point_block_size * bal_problem->point_index()[i];

        // cost_function: e
        // loss_function: H(e) 鲁棒核函数
        problem->AddResidualBlock(cost_function, loss_function, camera, point);
    }

}

// 求解优化问题
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
    srand(params.random_seed);   // 初始化随机数发生器
    bal_problem.Normalize();
    bal_problem.Perturb(params.rotation_sigma, params.translation_sigma,
                        params.point_sigma);

    std::cout << "Normalization complete..." << std::endl;
    
    Problem problem;
    BuildProblem(&bal_problem, &problem, params);

    std::cout << "the problem is successfully build.." << std::endl;
   
   
    Solver::Options options;
    SetSolverOptionsFromFlags(&bal_problem, params, &options);
    options.gradient_tolerance = 1e-16;
    options.function_tolerance = 1e-16;
    Solver::Summary summary;   // 统计函数每一次迭代求解的结果
    Solve(options, &problem, &summary);   // ceres求解功能
    std::cout << summary.FullReport() << "\n";

    // write the result into a .ply file.   
    if(!params.final_ply.empty()){
        bal_problem.WriteToPLYFile(params.final_ply);  // pay attention to this: ceres doesn't copy the value into optimizer, but implement on raw data! 
    }
}

int main(int argc, char** argv)
{    
    BundleParams params(argc,argv);  // set the parameters here.
   
    google::InitGoogleLogging(argv[0]);   // 初始化glog库，参数是第一个命令行参数即程序名，日志记录程序
    std::cout << params.input << std::endl;
    if(params.input.empty()){
        std::cout << "Usage: bundle_adjuster -input <path for dataset>";
        return 1;
    }

    SolveProblem(params.input.c_str(), params);
 
    return 0;
}
