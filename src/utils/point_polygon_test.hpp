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
    enum class FilterLocation : int {
        Center,
        LeftCenter,
        RightCenter,
        TopCenter,
        BottomCenter,
        LeftTop,
        RightTop,
        LeftBottom,
        RightBottom
    };

    /**
     * @brief box 多边形测试
     * @param box YOLO 检测框
     * @param polygon 顶点集合
     * @param filter_location 筛选位置
     * @param measureDist 是否返回距离
     * @return 拓扑位置 (-1: 在外部，0: 在边界，1: 在内部)
     */
    double boxPolygonTest(
        const Global::YoloDetectBox &box,
        const std::vector<cv::Point> &polygon,
        const FilterLocation filter_location = FilterLocation::Center,
        bool measureDist = false);

    /**
     * @brief 在多边形区域内过滤 YOLO 检测框
     * @param boxes YOLO 检测框的集合 (std::vector<Global::YoloDetectBox>)
     * @param polygon 表示多边形顶点的集合 (std::vector<cv::Point>)
     * @param inside 是否只保留多边形内部的检测框 (bool，默认为 true)
     * @param filter_location 筛选位置 (FilterLocation，默认为 FilterLocation::Center)
     * @return 在多边形内的检测框 index 集合 (std::vector<int>)
     */
    std::vector<int> filter_box_ids_by_polygon(
        const std::vector<Global::YoloDetectBox> &boxes,
        const std::vector<cv::Point> &polygon,
        const bool inside = true,
        const FilterLocation filter_location = FilterLocation::Center);

    /**
     * @brief 在多边形区域内过滤 YOLO 检测框
     * @param boxes YOLO 检测框的集合 (std::vector<Global::YoloDetectBox>)
     * @param polygon 表示多边形顶点的集合 (std::vector<cv::Point>)
     * @param inside 是否只保留多边形内部的检测框 (bool，默认为 true)
     * @param filter_location 筛选位置 (FilterLocation，默认为 FilterLocation::Center)
     * @return 在多边形内的检测框集合 (std::vector<Global::YoloDetectBox>)
     */
    std::vector<Global::YoloDetectBox> filter_boxes_by_polygon(
        const std::vector<Global::YoloDetectBox> &boxes,
        const std::vector<cv::Point> &polygon,
        const bool inside = true,
        const FilterLocation filter_location = FilterLocation::Center);

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

    void test_filter_boxes_by_polygon();

}

#endif // POINT_POLYGON_TEST_HPP
