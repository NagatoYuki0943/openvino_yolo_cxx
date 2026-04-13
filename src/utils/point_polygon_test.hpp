#ifndef POINT_POLYGON_TEST_HPP
#define POINT_POLYGON_TEST_HPP
#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "../global_vars.hpp"

namespace detect_utils
{

    /**
     * @brief 在多边形区域内过滤 YOLO 检测框
     * @param boxes YOLO 检测框的集合 (std::vector<Global::YoloDetectBox>)
     * @param polygon 表示多边形顶点的集合 (std::vector<cv::Point>)
     * @return 在多边形内的检测框 index 集合 (std::vector<int>)
     */
    std::vector<int> filter_box_ids_in_polygon(
        const std::vector<Global::YoloDetectBox> &boxes,
        const std::vector<cv::Point> &polygon);

    /**
     * @brief 在多边形区域内过滤 YOLO 检测框
     * @param boxes YOLO 检测框的集合 (std::vector<Global::YoloDetectBox>)
     * @param polygon 表示多边形顶点的集合 (std::vector<cv::Point>)
     * @return 在多边形内的检测框集合 (std::vector<Global::YoloDetectBox>)
     */
    std::vector<Global::YoloDetectBox> filter_boxes_in_polygon(
        const std::vector<Global::YoloDetectBox> &boxes,
        const std::vector<cv::Point> &polygon);

    /**
     * @brief 在图像上绘制封闭的多边形线条
     * @param image 要在其上绘制的图像 (cv::Mat)
     * @param polygon_points 多边形的顶点集合 (std::vector<cv::Point>)
     * @param color 线条颜色 (cv::Scalar，例如 cv::Scalar(0, 255, 0) 为绿色)
     * @param thickness 线条粗细 (int，默认为 2)
     */
    void draw_closed_polygon(
        cv::Mat &image,
        const std::vector<cv::Point> &polygon_points,
        const cv::Scalar &color = {0, 255, 255},
        const int thickness = 2);

    void test_filter_boxes_in_polygon();

}

#endif // POINT_POLYGON_TEST_HPP
