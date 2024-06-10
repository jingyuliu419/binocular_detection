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

    // // 计算中心点
    cv::Point center(0, 0);
    for (const auto& point : boundingPoints) {
        center += point;
    }
    center.x /= boundingPoints.size();
    center.y /= boundingPoints.size();

    // 计算中心点
    // center.x = image.cols *0.5;
    // center.y = image.rows *0.5;

    // 进行边缘检测
    cv::Mat edges;
    Canny(image, edges, 0, 3000, 7);

    // 进行霍夫直线变换
    std::vector<cv::Vec2f> lines;
    HoughLines(edges, lines, 1, CV_PI / 180, 40);


    // 绘制检测到的直线
    for (size_t i = 0; i < lines.size(); ++i) {
        float rho = lines[i][0];
        float theta = lines[i][1];
        cv::Point pt1, pt2;

        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;

        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));

        line(image, pt1, pt2, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }

    // 存储直线交点
    std::vector<cv::Point2f> intersections;

    // 检查直线交点
    if (lines.size() >= 2) {
        std::cout << "lines in the image: " << lines.size() << std::endl;
        // 计算直线交点
        for (size_t i = 0; i < lines.size(); ++i) {
            for (size_t j = i + 1; j < lines.size(); ++j) {
                float rho1 = lines[i][0], theta1 = lines[i][1];
                float rho2 = lines[j][0], theta2 = lines[j][1];

                float angleDiff = abs(theta1 - theta2);
                while(angleDiff > CV_PI)
                    angleDiff = CV_PI - angleDiff;
                if(angleDiff < CV_PI*0.4)
                    continue;

                double A = cos(theta1), B = sin(theta1);
                double C = cos(theta2), D = sin(theta2);
                double det = A * D - B * C;

                if (det != 0) {
                    cv::Point2f intersection;
                    intersection.x = (D * rho1 - B * rho2) / det;
                    intersection.y = (-C * rho1 + A * rho2) / det;

                    intersections.push_back(intersection);
                }
            }
        }

    } else {
        std::cout << "Could find " << lines.size() << " lines in the image." << std::endl;
    }

    // 中心交点
    cv::Point cornerCenter = center;

    for (const auto& intersection : intersections) {
        cv::circle(image, intersection, 1, cv::Scalar(255, 0, 0), -1);//blue
        // 中心交点
        int distance = INT_MAX;
        int newDistance = sqrt( pow((center.x - intersection.x), 2) + pow((center.y - intersection.y), 2));
        if(distance > newDistance){
            distance = newDistance;
            cornerCenter = intersection;
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