#ifndef DETECT_UTILS_HPP
#define DETECT_UTILS_HPP
#pragma once

#include <string>
#include <vector>
#include <map>
#include "../global_vars.hpp"


namespace detect_utils
{

    // 辅助函数：根据固定的 ID 生成稳定且明亮的颜色
    cv::Scalar GetColorForId(const int id);

    void draw_detected_object(cv::Mat &image, const std::vector<Global::YoloDetectBox> &detect_boxes);

    /**
     * @brief 根据类别 ID 将检测结果进行分类
     * @param detect_boxes 检测结果
     * @return 按类别 ID 分组的 box ID
     */
    std::map<int, std::vector<int>> classify_box_id_by_class(const std::vector<Global::YoloDetectBox> &detect_boxes);

    /**
     * @brief 根据类别 ID 将检测结果进行分类
     * @param detect_boxes 检测结果
     * @return 按类别 ID 分组的检测结果
     */
    std::map<int, std::vector<Global::YoloDetectBox>> classify_box_by_class(const std::vector<Global::YoloDetectBox> &detect_boxes);

    /**
     * @brief 合并同类别检测结果
     * @param classified_boxes 按类别 ID 分组的检测结果
     * @return 合并后的检测结果
     */
    std::vector<Global::YoloDetectBox> merge_classified_boxes(const std::map<int, std::vector<Global::YoloDetectBox>> &classified_boxes);

    /**
     * @brief 计算两个 Box 之间的 IoU (Intersection over Union)
     */
    float calculate_iou(const Global::YoloDetectBox &box1, const Global::YoloDetectBox &box2);

    /**
     * @brief 计算两个 Box 之间的 IoA (Intersection over Area)
     */
    float calculate_ioa(const Global::YoloDetectBox &ref_box, const Global::YoloDetectBox &target_box);

    /**
     * @brief 过滤 target_boxes 中与 reference_boxes 重叠度大于 threshold 的目标
     * @param target_boxes 待过滤的目标列表
     * @param reference_boxes 作为参考的目标列表
     * @param threshold 重叠度阈值，大于此值的 target_box 会被忽略
     * @param use_ioa 是否使用 IoA 计算重叠度
     * @return 过滤后的检测结果
     */
    std::vector<Global::YoloDetectBox> filter_boxes_by_reference(
        const std::vector<Global::YoloDetectBox> &target_boxes,
        const std::vector<Global::YoloDetectBox> &reference_boxes,
        float threshold = 0.7f,
        bool use_ioa = true);

    /**
     * @brief 过滤掉某些类别的检测结果
     * @param detect_boxes 检测结果
     * @param target_id 要过滤的类别 ID
     * @param ref_id 参考类别 ID
     * @param threshold 重叠度阈值，大于此值的 target_box 会被忽略
     * @param use_ioa 是否使用 IoA 计算重叠度
     * @return 过滤后的检测结果
     */
    std::vector<Global::YoloDetectBox> filter_target_on_ref(
        const std::vector<Global::YoloDetectBox> &detect_boxes,
        int target_id,
        int ref_id,
        float threshold = 0.7f,
        bool use_ioa = true);

}

#endif // DETECT_UTILS_HPP
