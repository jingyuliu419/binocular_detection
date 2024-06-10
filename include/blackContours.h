#ifndef shi_tomasi_h
#define shi_tomasi_h

#include <iostream>
#include <string>
#include <stdio.h>
#include <unistd.h>
#include <thread>
#include <condition_variable>
#include <opencv2/opencv.hpp>

class FindRectCenter
{
public:
    cv::Point operator()(const cv::Mat& image) const {
    // 不要cv::COLOR_BGR2YGRA, gray不能cv::COLOR_BGR2HSV
    // cv::cvtColor(image, image, cv::COLOR_BGR2YGRA);

    // 增强对比度
    {
        // double min_val, max_val;
        // cv::minMaxLoc(image, &min_val, &max_val);
        // // 线性变换 增强对比度
        // image.convertTo(image, -1, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));
        
        // double alpha = 2.0; // 调整对比度参数
        // int beta = 0; // 调整亮度参数
        // image.convertTo(image, -1, alpha, beta);
    } 

    // 转换图像为HSV颜色空间
    // H——Hue即色相,其中红色对应0度，绿色对应120度，蓝色对应240度。
    // S——Saturation即饱和度，色彩的深浅度(0-100%) 
    // V——Value即色调，纯度，色彩的亮度(0-100%)
    // 直接使用OpenCV中cvtColor函数，并设置参数为CV_BGR2HSV
    // 那么所得的H、S、V值范围分别是[0,180)，[0,255)，[0,255)，而非[0,360]，[0,1]，[0,1]；
    cv::Mat hsvImage;
    cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

    // 定义黑色区域的HSV阈值范围(白天:black  黑夜:white)
    cv::Scalar lower_black = cv::Scalar(0, 0, 0);// black(0-180)(0-255)(0-46)
    cv::Scalar upper_black = cv::Scalar(180, 255, 46);
    // cv::Scalar lower_black = cv::Scalar(0, 0, 221);// white(0-180)(0-30)(221-255)
    // cv::Scalar upper_black = cv::Scalar(180, 30, 255);

    // 根据阈值进行颜色分割，提取黑色区域
    cv::Mat mask;
    cv::inRange(hsvImage, lower_black, upper_black, mask);

    // 中心交点
    cv::Point cornerCenter;
    //如果没有找到就用目标检测的中心
    if(mask.empty()){
        std::cout << "没有找到黑色区域!!" << std::endl;
        cornerCenter = cv::Point(0.5*image.cols, 0.5*image.rows);
        return cornerCenter;
    }

    // 寻找黑色区域的轮廓 和 坐标点
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);//在找轮廓

    // 找出最大的两个轮廓
    std::vector<std::vector<cv::Point>> largestContours;
    size_t numContours = contours.size();
    cv::Point closestPoint1, closestPoint2;// 两个轮廓中的所有点对中,最近的两个点
    if (numContours >= 2) {
        // cout << "轮廓数量:" << numContours << endl;
        // 使用 lambda 函数按轮廓面积降序排序
        sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
            return cv::contourArea(c1) > cv::contourArea(c2);
        });

        largestContours.push_back(contours[0]);
        largestContours.push_back(contours[1]);

        // 绘制黑色区域的轮廓
        cv::Scalar color(0, 255, 0); // 绿色轮廓
        for (size_t i = 0; i < largestContours.size(); i++) {
            // cout << "轮廓" << i+1 << "的点数量: " << largestContours[i].size() << endl;
            cv::drawContours(image, largestContours, static_cast<int>(i), color, -1);
        }

        // 初始化最小距离及其索引
        double minDistance = DBL_MAX;

        // 遍历两个轮廓中的所有点对，找到最近的两个点
        for (const auto& point1 : largestContours[0]) {
            for (const auto& point2 : largestContours[1]) {
                double distance = norm(point1 - point2);
                if (distance < minDistance) {
                    minDistance = distance;
                    closestPoint1 = point1;
                    closestPoint2 = point2;
                }
            }
        }


        // 计算两个最近点的中间点
        cornerCenter = (closestPoint1 + closestPoint2) / 2;
        cv::circle(image, cornerCenter, 2, cv::Scalar(0, 0, 255), -1);//red -1:filled

        // 输出最近点和中间点的坐标
        // cout << "最近点1坐标: " << closestPoint1 << endl;
        // cout << "最近点2坐标: " << closestPoint2 << endl;
    } else {
        cornerCenter = cv::Point(0.5*image.cols, 0.5*image.rows);
        std::cout << "轮廓数量" << numContours << ", 不足两个，无法找到最大的两个轮廓" << std::endl;
    }
    // 计算中心点
    // cornerCenter.x = image.cols *0.5;
    // cornerCenter.y = image.rows *0.5;
    return cornerCenter;
    }
};

#endif