#include <string>
#include <vector>
#include <map>
#include "../global_vars.hpp"
#include "functions.hpp"

namespace detect_utils
{

    // 辅助函数：根据固定的 ID 生成稳定且明亮的颜色
    cv::Scalar GetColorForId(const int id)
    {
        // 使用质数进行简单的哈希计算，放大不同 ID 之间的颜色差异
        // 取模 136 然后加上 120，确保颜色通道值在 120-255 之间（保证明亮度）
        int r = 120 + ((id * 37) % 136);
        int g = 120 + ((id * 73) % 136);
        int b = 120 + ((id * 109) % 136);
        return cv::Scalar(b, g, r);
    }

    void draw_detected_object(cv::Mat &image, const std::vector<Global::YoloDetectBox> &detect_boxes)
    {
        for (const auto &box : detect_boxes)
        {
            const float &confidence = box.confidence;
            const std::string &class_name = box.class_name;
            const int &class_id = box.class_id;

            // 优先使用 track_id 获取颜色。如果未使用追踪器，则退化为按类别区分颜色
            int id_to_color = (box.track_id > 0) ? box.track_id : class_id;
            const cv::Scalar color = GetColorForId(id_to_color);

            // 绘制目标边界框
            cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 2);

            // 准备标签文本
            std::string classString = class_name + " " + std::to_string(confidence).substr(0, 4);
            if (box.track_id > 0)
            {
                classString += " ID:" + std::to_string(box.track_id);
            }

            // 计算文本框大小
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 0.75, 1, 0);

            // 动态计算文本框和文字的 Y 坐标，防止物体在图像最顶端时文字出界
            int text_box_top, text_top;
            if (box.top > textSize.height + 5)
            {
                text_box_top = box.top - textSize.height - 10;
                text_top = box.top - 5;
            }
            else
            {
                // 如果物体太靠上，把文字框画在边界框内部
                text_box_top = box.top;
                text_top = box.top + textSize.height + 5;
            }

            // 绘制文本背景框 (FILLED)
            cv::rectangle(image, {box.left, text_box_top}, {box.left + textSize.width, text_box_top + textSize.height + 10}, color, cv::FILLED);

            // 绘制文本（使用黑色字体保证在明亮背景上的对比度）
            cv::putText(image, classString, {box.left, text_top}, cv::FONT_HERSHEY_DUPLEX, 0.75, cv::Scalar(0, 0, 0), 1, 0);
        }
    }

    /**
     * @brief 根据类别 ID 将检测结果进行分类
     * @param detect_boxes 检测结果
     * @return 按类别 ID 分组的 box ID
     */
    std::map<int, std::vector<int>> classify_box_id_by_class(const std::vector<Global::YoloDetectBox> &detect_boxes)
    {
        std::map<int, std::vector<int>> class_map;
        for (int i = 0; i < detect_boxes.size(); i++)
        {
            class_map[detect_boxes[i].class_id].push_back(i);
        }
        return class_map;
    }

    /**
     * @brief 根据类别 ID 将检测结果进行分类
     * @param detect_boxes 检测结果
     * @return 按类别 ID 分组的检测结果
     */
    std::map<int, std::vector<Global::YoloDetectBox>> classify_box_by_class(const std::vector<Global::YoloDetectBox> &detect_boxes)
    {
        std::map<int, std::vector<Global::YoloDetectBox>> classified_boxes;
        for (const auto &box : detect_boxes)
        {
            classified_boxes[box.class_id].push_back(box);
        }
        return classified_boxes;
    }

    /**
     * @brief 合并同类别检测结果
     * @param classified_boxes 按类别 ID 分组的检测结果
     * @return 合并后的检测结果
     */
    std::vector<Global::YoloDetectBox> merge_classified_boxes(const std::map<int, std::vector<Global::YoloDetectBox>> &classified_boxes)
    {
        std::vector<Global::YoloDetectBox> merged_boxes;
        int total_size = 0;
        for (auto const &[id, boxes] : classified_boxes)
            total_size += boxes.size();
        merged_boxes.reserve(total_size);

        for (const auto &[class_id, boxes] : classified_boxes)
        {
            merged_boxes.insert(merged_boxes.end(), boxes.begin(), boxes.end());
        }
        return merged_boxes;
    }

    /**
     * @brief 计算两个 Box 之间的 IoU (Intersection over Union)
     */
    float calculate_iou(const Global::YoloDetectBox &box1, const Global::YoloDetectBox &box2)
    {
        // 计算交集的左上角和右下角坐标
        int inter_left = std::max(box1.left, box2.left);
        int inter_top = std::max(box1.top, box2.top);
        int inter_right = std::min(box1.right, box2.right);
        int inter_bottom = std::min(box1.bottom, box2.bottom);

        // 如果没有交集，宽或高会小于等于 0
        int inter_width = std::max(0, inter_right - inter_left);
        int inter_height = std::max(0, inter_bottom - inter_top);

        // 交集面积
        int inter_area = inter_width * inter_height;

        // 两个 Box 各自的面积
        int area1 = (box1.right - box1.left) * (box1.bottom - box1.top);
        int area2 = (box2.right - box2.left) * (box2.bottom - box2.top);

        // 并集面积 = 面积1 + 面积2 - 交集面积
        int union_area = area1 + area2 - inter_area;

        // 防止除以 0
        if (union_area <= 0)
            return 0.0f;

        return static_cast<float>(inter_area) / static_cast<float>(union_area);
    }

    /**
     * @brief 计算两个 Box 之间的 IoA (Intersection over Area)
     */
    float calculate_ioa(const Global::YoloDetectBox &ref_box, const Global::YoloDetectBox &target_box)
    {
        int inter_left = std::max(ref_box.left, target_box.left);
        int inter_top = std::max(ref_box.top, target_box.top);
        int inter_right = std::min(ref_box.right, target_box.right);
        int inter_bottom = std::min(ref_box.bottom, target_box.bottom);

        int inter_width = std::max(0, inter_right - inter_left);
        int inter_height = std::max(0, inter_bottom - inter_top);
        int inter_area = inter_width * inter_height;

        // 只计算被过滤目标 (target_box) 的自身面积
        int target_area = (target_box.right - target_box.left) * (target_box.bottom - target_box.top);

        if (target_area <= 0)
            return 0.0f;

        // 返回：交集面积 占据 目标区域面积 的百分比
        return static_cast<float>(inter_area) / static_cast<float>(target_area);
    }

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
        float threshold,
        bool use_ioa)
    {
        std::vector<Global::YoloDetectBox> filtered_boxes;
        filtered_boxes.reserve(target_boxes.size());

        for (const auto &target_box : target_boxes)
        {
            bool should_save = true;

            // 拿当前 target_box 去和所有的 reference_box 比较
            for (const auto &ref_box : reference_boxes)
            {
                float iou_or_ioa;
                if (use_ioa)
                {
                    iou_or_ioa = calculate_ioa(ref_box, target_box);
                }
                else
                {
                    iou_or_ioa = calculate_iou(ref_box, target_box);
                }

                if (iou_or_ioa > threshold)
                {
                    should_save = false;
                    break; // 只要和其中一个重叠度超标，就忽略它，不用再往后比了
                }
            }

            // 如果没有被忽略，则保留下来
            if (should_save)
            {
                filtered_boxes.push_back(target_box);
            }
        }

        return filtered_boxes;
    }

    /**
     * @brief 通过一个类别过滤掉另一个类别
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
        float threshold,
        bool use_ioa)
    {
        if (detect_boxes.size() < 2)
            return detect_boxes;

        if (target_id == ref_id)
            return detect_boxes;

        auto classified_boxes = classify_box_by_class(detect_boxes);

        auto it_target = classified_boxes.find(target_id);
        auto it_ref = classified_boxes.find(ref_id);

        if (it_target != classified_boxes.end() && it_ref != classified_boxes.end())
        {
            it_target->second = filter_boxes_by_reference(it_target->second, it_ref->second, threshold, use_ioa);
        }

        return merge_classified_boxes(classified_boxes);
    }

}
