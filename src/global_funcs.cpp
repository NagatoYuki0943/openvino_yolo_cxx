#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include "global_vars.hpp"
#include "global_funcs.hpp"


namespace Global
{

    GereralConfig read_config(const std::string &config_path)
    {
        GereralConfig config = {};

        try
        {
            nlohmann::json j;
            std::ifstream ifs(config_path);
            ifs >> j;
            ifs.close();

            if (j.contains("model_path"))
                config.detect_config.model_path = j["model_path"].get<std::string>();
            if (j.contains("imgsz"))
            {
                // [height, width]
                std::vector<int> imgsz = j["imgsz"].get<std::vector<int>>();
                if (imgsz.size() == 2)
                {
                    config.detect_config.model_input_shape.width = imgsz[1];
                    config.detect_config.model_input_shape.height = imgsz[0];
                }
            }
            if (j.contains("conf_threshold"))
                config.detect_config.conf_threshold = j["conf_threshold"].get<float>();
            if (j.contains("nms_threshold"))
                config.detect_config.nms_threshold = j["nms_threshold"].get<float>();
            if (j.contains("names"))
            {
                config.detect_config.classes.clear();
                for (auto &item : j["names"].items())
                {
                    int id = std::stoi(item.key());
                    std::string name = item.value().get<std::string>();
                    config.detect_config.classes[id] = name;
                }
            }

            if (j.contains("track"))
            {
                auto t = j["track"];
                if (t.contains("max_time_lost"))
                    config.track_config.max_time_lost = t["max_time_lost"].get<int>();
                if (t.contains("track_high_thresh"))
                    config.track_config.track_high_thresh = t["track_high_thresh"].get<float>();
                if (t.contains("track_low_thresh"))
                    config.track_config.track_low_thresh = t["track_low_thresh"].get<float>();
                if (t.contains("new_track_thresh"))
                    config.track_config.new_track_thresh = t["new_track_thresh"].get<float>();
                if (t.contains("match_thresh"))
                    config.track_config.match_thresh = t["match_thresh"].get<float>();
            }

            return config;
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            return config;
        }
    }

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

    void draw_detected_object(cv::Mat &image, const std::vector<YoloDetectBox> &detect_boxes)
    {
        for (const auto &box : detect_boxes)
        {
            const float &confidence = box.confidence;
            const std::string &class_name = box.class_name;
            const int &class_id = box.class_id;

            // 优先使用 track_id 获取颜色。如果未使用追踪器，则退化为按类别区分颜色
            int id_to_color = (box.track_id >= 0) ? box.track_id : class_id;
            const cv::Scalar color = GetColorForId(id_to_color);

            // 绘制目标边界框
            cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 2);

            // 准备标签文本
            std::string classString = class_name + " " + std::to_string(confidence).substr(0, 4);
            if (box.track_id >= 0)
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

    std::map<int, std::vector<int>> classify_boxed_by_class(const std::vector<YoloDetectBox> &detect_boxes)
    {
        std::map<int, std::vector<int>> class_map;
        for (int i = 0; i < detect_boxes.size(); i++)
        {
            class_map[detect_boxes[i].class_id].push_back(i);
        }
        return class_map;
    }
}
