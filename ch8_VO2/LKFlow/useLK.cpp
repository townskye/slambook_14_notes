#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

int main( int argc, char** argv )
{
    if ( argc != 2 )
    {
        cout<<"usage: useLK path_to_dataset"<<endl;
        return 1;
    }
    string path_to_dataset = argv[1];
    string associate_file = path_to_dataset + "/associate.txt";
    
    // 打开文件并写入fin
    ifstream fin( associate_file );
    if ( !fin ) 
    {
        cerr<<"I cann't find associate.txt!"<<endl;
        return 1;
    }
    
    string rgb_file, depth_file, time_rgb, time_depth;
    list< cv::Point2f > keypoints;      // 因为要删除跟踪失败的点，使用list
    cv::Mat color, depth, last_color;
    
    for ( int index=0; index<100; index++ )
    {
        // 从输入流 fin 中读取文件名和地址，每次只读一行（四个元素），因为fin后只接了四个参数
        fin>>time_rgb>>rgb_file>>time_depth>>depth_file;
        color = cv::imread( path_to_dataset+"/"+rgb_file );
	// imread第二个参数表示读取图像的色彩格式，-1表示深度图
        depth = cv::imread( path_to_dataset+"/"+depth_file, -1 );
        if (index ==0 )
        {
            // 对第一帧提取FAST特征点
            vector<cv::KeyPoint> kps;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect( color, kps );
            for ( auto kp:kps )
	        // 获取所提取特征点的坐标
                keypoints.push_back( kp.pt );
            last_color = color;
            continue;
        }
        // 无对应图片时直接进行新的循环
        if ( color.data==nullptr || depth.data==nullptr )
            continue;
        // 对其他帧用LK跟踪特征点
        vector<cv::Point2f> next_keypoints;
        vector<cv::Point2f> prev_keypoints;
	// vector是模板（构造对象）而非类型，vector不能用vector进行赋值，但可以用来初始化
	// 这里keypoints是list, 因此必须使用循环赋给prev_keypoints
        for ( auto kp:keypoints )
            prev_keypoints.push_back(kp);
        vector<unsigned char> status;
        vector<float> error; 
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	// status:状态向量，如果找到相应特征的流，则向量的每个元素设置为1，否则为0
	// error:原始图像碎片与找到的移动点之间的误差
        cv::calcOpticalFlowPyrLK( last_color, color, prev_keypoints, next_keypoints, status, error );
	// next_keypoints 的维数等于 prev_keypoints 的维数
	// cout << "size 1: " << prev_keypoints.size() << " " << "size 2: " << next_keypoints.size() << endl;
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
        cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;
        // 把跟丢的点删掉，并更新keypoints为next_keypoints，供下一步的光流匹配使用
        int i=0; 
        for ( auto iter=keypoints.begin(); iter!=keypoints.end(); i++)
        {
	    // 删掉跟丢的点
            if ( status[i] == 0 )
            {
                // list的erase操作
	        iter = keypoints.erase(iter);
                continue;
            }
            // 未跟丢的点，更新为新一帧图像中的坐标
            *iter = next_keypoints[i];
            iter++;
        }
        cout<<"tracked keypoints: "<<keypoints.size()<<endl;
        if (keypoints.size() == 0)
        {
            cout<<"all keypoints are lost."<<endl;
            break; 
        }
        // 画出 keypoints
        cv::Mat img_show = color.clone();
        for ( auto kp:keypoints )
            cv::circle(img_show, kp, 10, cv::Scalar(0, 240, 0), 1);
        cv::imshow("corners", img_show);
        cv::waitKey(0);
	// 将当前的第二帧图像设为新的参考图像，继续进行新的光流跟踪
        last_color = color;
    }
    return 0;
}