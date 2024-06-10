#ifndef depthComputer_h
#define depthComputer_h

#include <iostream>
#include <string>
#include <stdio.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class depthCompute
{
public:
    depthCompute(int x, int y);//初始化
    void De_distortion(cv::Mat &src, cv::Mat &dst);//去畸变的
    void depth(cv::Mat &dst);
    int compressDepth();
    //左相机mark中心点坐标
    int x;
    int y;
    //右相机mark中心点坐标
    int xr;
    int yr;

    cv::Mat imgDistortedL;//去畸变后的图像，用于检测
    cv::Mat imgDistortedR;//去畸变后的图像，用于检测

    void get_compressBaseline();
    void computeCompress();//次数和频率(暂无深度)
    std::pair<int16_t, int16_t> getCompress();//次数和频率(暂无深度)
    void clear_compress();   

private:
    cv::Mat P1, P2;
    
    //坐标点在相机坐标系下的Y轴位置，单位：mm
    int Y;

    //右相机相对于左相机的平移矩阵和旋转矩阵 matlab的互为转置矩阵
    const cv::Mat R = (cv::Mat_<double>(3, 3) << 1.0000, 0.000592, -0.0063,
                       -0.0005616, 1.0000, 0.0049,
                       0.0063, -0.0048, 1.0000);
    const cv::Mat T = (cv::Mat_<double>(3, 1) << -166.5988, -0.7989, -2.3029);
    // const cv::Mat R = (cv::Mat_<double>(3, 3) << 1.0000, 0.0016, 0.0020,
    //                 -0.0016, 1.0000, 0.0047,
    //                 -0.0020, -0.0047, 1.0000);
    // const cv::Mat T = (cv::Mat_<double>(3, 1) << -166.0000, -0.6296, 0.1965);

    // 相机内参，matlab标定，竖着填写
    // L
    const cv::Mat cameraMatrixL = (cv::Mat_<double>(3, 3) << 1178.4, 0.0, 766.1886,
                                   0.0, 1177.6, 445.7682,
                                   0.0, 0.0, 1.0);
    const cv::Mat distCoeffsL = (cv::Mat_<double>(5, 1) << 0.0086, 0.0337, -0.00040448, 0.0045, -0.1853); // k1 k2 p1 p2 k3
    // const cv::Mat cameraMatrixL = (cv::Mat_<double>(3, 3) << 1181.2, 0.0, 746.7586,
    //                                0.0, 1181.6, 451.3456,
    //                                0.0, 0.0, 1.0);
    // const cv::Mat distCoeffsL = (cv::Mat_<double>(5, 1) << 0.0119, 0.0979, -0.00024, -0.00069, -0.6606); // k1 k2 p1 p2 k3

    
    // R
    const cv::Mat cameraMatrixR = (cv::Mat_<double>(3, 3) << 1179.7, 0.0, 703.5117,
                                   0.0, 1177.6, 429.6743,
                                   0.0, 0.0, 1.0);
    const cv::Mat distCoeffsR = (cv::Mat_<double>(5, 1) << 0.0281, -0.0812, -0.0013, 0.0055, 0.0464); // k1 k2 p1 p2 k3 
    // const cv::Mat cameraMatrixR = (cv::Mat_<double>(3, 3) << 1181.0, 0.0, 695.9637,
    //                                0.0, 1181.2, 432.5191,
    //                                0.0, 0.0, 1.0);
    // const cv::Mat distCoeffsR = (cv::Mat_<double>(5, 1) << 0.0327, -0.0601, -0.00003247, -0.00088192, -0.0092); // k1 k2 p1 p2 k3                           
    
    std::queue<int32_t> compressTrack;//按压轨迹 size>50删除队头
    int16_t compressNumber = 0;//按压次数
    int32_t compressBaseline = 0;//按压基准
    int16_t compressFrequency = 0;//按压频率
    std::chrono::milliseconds lastTime;//按压频率lastTime
    std::chrono::milliseconds currentTime;//按压频率currentTime    
};
#endif