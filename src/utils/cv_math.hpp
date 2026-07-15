#ifndef CV_MATH_HPP
#define CV_MATH_HPP
#pragma once

#include <opencv2/opencv.hpp>

namespace detect_utils
{
    /**
     * @brief 这个函数计算的是由三个二维点 a、b、c 构成的两个向量 AB 和 AC 的二维叉乘（Cross Product）结果（即叉积在Z轴方向的标量值）。
     * @param a 第一个点
     * @param b 第二个点
     * @param c 第三个点
     * @return 返回叉积在Z轴方向的标量值（double类型）
     */
    double cross(const cv::Point2d &a, const cv::Point2d &b, const cv::Point2d &c);

    /**
     * @brief 判断点 p 是否严格位于由点 ab 上
     * @param a 线段的起点
     * @param b 线段的终点
     * @param p 要判断的点
     * @return 返回点 p 是否在线段 ab 上（bool类型）
     */
    bool on_segment(const cv::Point2d &a, const cv::Point2d &b, const cv::Point2d &p);

    /**
     * @brief 判断两条线段是否相交
     *        核心思想是：如果两条线段相交，那么其中一条线段的两个端点，必然分布在另一条线段所在直线的两侧。
     * @param p1 线段1的起点
     * @param p2 线段1的终点
     * @param q1 线段2的起点
     * @param q2 线段2的终点
     * @return 返回两条线段是否相交（bool类型）
     */
    bool segments_intersect(
        const cv::Point2d &p1,
        const cv::Point2d &p2,
        const cv::Point2d &q1,
        const cv::Point2d &q2);

    enum class CoordSystem
    {
        OpenCV,
        Math
    };

    /**
     * @brief 计算两个点之间的角度
     * @param start 起始点
     * @param end 终止点
     * @param coord_system 坐标系类型（OpenCV或Math）
     * @return 返回角度（度）
     */
    double calc_line_angle(
        const cv::Point2d &start,
        const cv::Point2d &end,
        const CoordSystem coord_system = CoordSystem::OpenCV);

    void test_segments_intersect();

    void test_calc_line_angle();
}

#endif // CV_MATH_HPP
