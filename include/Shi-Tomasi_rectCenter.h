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
        // 查找轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 筛选正方形轮廓
        std::vector<std::vector<cv::Point>> squareContours;
        for (const auto& contour : contours) {
            double epsilon = 0.02 * cv::arcLength(contour, true);
            std::vector<cv::Point> approx;
            cv::approxPolyDP(contour, approx, epsilon, true);

            if (approx.size() == 4) {
                squareContours.push_back(approx);
            }
        }

        // 找出大正方形的边界点
        std::vector<cv::Point> boundingPoints;
        for (const auto& squareContour : squareContours) {
            for (const auto& point : squareContour) {
                boundingPoints.push_back(point);
            }
        }

        // 计算中心点
        cv::Point center(0, 0);
        for (const auto& point : boundingPoints) {
            center += point;
        }
        center.x /= boundingPoints.size();
        center.y /= boundingPoints.size();

        // // 计算中心点
        // center.x = image.cols *0.5;
        // center.y = image.rows *0.5;

        // 使用Shi-Tomasi角点检测
        std::vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(image, corners, 100, 0.01, 10);

        // 绘制角点
        for (const auto& corner : corners) {
            cv::circle(image, corner, 0.5, cv::Scalar(0, 0, 255), -1);//red
        }

        // 中心角点
        cv::Point cornerCenter(0, 0);
        int distance = INT_MAX;
        for (const auto& corner : corners) {
            int newDistance = sqrt( pow((corner.x - center.x), 2) + pow((corner.y - center.y), 2));
            if(distance > newDistance){
                distance = newDistance;
                cornerCenter = corner;
            }
        }

        // 在图像上标注中心点
        cv::circle(image, cornerCenter, 0.5, cv::Scalar(0, 255, 0), 2);//green

        // 显示图像
        // int dsize = 4;d
        // cv::resize(image, image, cv::Size(dsize*image.cols, dsize*image.rows), 0, 0, cv::INTER_AREA);
        // cv::imshow("Square Center", image);

        return cornerCenter;
    }
};

#endif