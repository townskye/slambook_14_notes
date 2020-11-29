/*
 * 空间点到像素点的转换
*/
#ifndef PROJECTION_H
#define PROJECTION_H

#include "tools/rotation.h"

// camera : 9 dims array with 
// [0-2] : angle-axis rotation 
// [3-5] : translation
// [6-8] : camera parameter, [6] focal length, [7-8] second and forth order radial distortion
// point : 3D location. 
// predictions : 2D predictions with center of the image plane. 

// 模板函数
template<typename T>
// 考虑畸变后的空间点到像素点的变换
inline bool CamProjectionWithDistortion(const T* camera, const T* point, T* predictions){
    // Rodrigues' formula
    // se(3)中的旋转就是变换矩阵中的旋转，但平移不是变换矩阵中的平移向量
    // so(3)就是由旋转向量组成的空间
    T p[3];
    // 这里camera是一个指向9维数据的指针，将指针传递给数组，数组名本身就是一个指针
    // 数组名包含的是指针指向的地址和数据类型
    // 因此可以用这种方式直接取相机参数的前三位数据
    AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5] are the translation
    // 空间点在相机坐标系的坐标
    p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];

    // Compute the center fo distortion
    // 负的归一化平面的坐标
    T xp = -p[0]/p[2];
    T yp = -p[1]/p[2];

    // Apply second and fourth order radial distortion
    const T& l1 = camera[7];
    const T& l2 = camera[8];

    T r2 = xp*xp + yp*yp;
    T distortion = T(1.0) + r2 * (l1 + l2 * r2);

    const T& focal = camera[6];
    predictions[0] = focal * distortion * xp;
    predictions[1] = focal * distortion * yp;

    return true;
}



#endif // projection.h