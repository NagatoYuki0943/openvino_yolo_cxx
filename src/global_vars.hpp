#ifndef GLOBAL_VARS_HPP
#define GLOBAL_VARS_HPP
#pragma once

#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

namespace Global
{
    struct YoloDetectBox
    {
        int class_id;               // 类别 id
        std::string class_name;     // 类别名称
        float confidence;           // 置信度
        int left;                   // 左上角 x 坐标
        int top;                    // 左上角 y 坐标
        int right;                  // 右下角 x 坐标
        int bottom;                 // 右下角 y 坐标
        std::uint64_t track_id = 0; // 追踪 id, 0 代表没有追踪
        std::uint64_t frame_id = 0; // 帧 id
    };

    inline std::map<int, std::string> default_classes = {
        {0, "person"},
        {1, "bicycle"},
        {2, "car"},
        {3, "motorcycle"},
        {4, "airplane"},
        {5, "bus"},
        {6, "train"},
        {7, "truck"},
        {8, "boat"},
        {9, "traffic light"},
        {10, "fire hydrant"},
        {11, "stop sign"},
        {12, "parking meter"},
        {13, "bench"},
        {14, "bird"},
        {15, "cat"},
        {16, "dog"},
        {17, "horse"},
        {18, "sheep"},
        {19, "cow"},
        {20, "elephant"},
        {21, "bear"},
        {22, "zebra"},
        {23, "giraffe"},
        {24, "backpack"},
        {25, "umbrella"},
        {26, "handbag"},
        {27, "tie"},
        {28, "suitcase"},
        {29, "frisbee"},
        {30, "skis"},
        {31, "snowboard"},
        {32, "sports ball"},
        {33, "kite"},
        {34, "baseball bat"},
        {35, "baseball glove"},
        {36, "skateboard"},
        {37, "surfboard"},
        {38, "tennis racket"},
        {39, "bottle"},
        {40, "wine glass"},
        {41, "cup"},
        {42, "fork"},
        {43, "knife"},
        {44, "spoon"},
        {45, "bowl"},
        {46, "banana"},
        {47, "apple"},
        {48, "sandwich"},
        {49, "orange"},
        {50, "broccoli"},
        {51, "carrot"},
        {52, "hot dog"},
        {53, "pizza"},
        {54, "donut"},
        {55, "cake"},
        {56, "chair"},
        {57, "couch"},
        {58, "potted plant"},
        {59, "bed"},
        {60, "dining table"},
        {61, "toilet"},
        {62, "tv"},
        {63, "laptop"},
        {64, "mouse"},
        {65, "remote"},
        {66, "keyboard"},
        {67, "cell phone"},
        {68, "microwave"},
        {69, "oven"},
        {70, "toaster"},
        {71, "sink"},
        {72, "refrigerator"},
        {73, "book"},
        {74, "clock"},
        {75, "vase"},
        {76, "scissors"},
        {77, "teddy bear"},
        {78, "hair drier"},
        {79, "toothbrush}"}};

    struct DetectConfig
    {
        std::string model_path = "";
        float conf_threshold = 0.25;
        float nms_threshold = 0.5;
        // [width, height]
        cv::Size model_input_shape = cv::Size(640, 640);
        std::map<int, std::string> classes = default_classes;
    };

    struct TrackConfig
    {
        int max_time_lost = 60;
        float track_high_thresh = 0.25;
        float track_low_thresh = 0.1;
        float new_track_thresh = 0.25;
        float match_thresh = 0.8;
        int min_hits = 1;
    };

    struct GereralConfig
    {
        DetectConfig detect_config;
        TrackConfig track_config;
    };
}

#endif // GLOBAL_VARS_HPP
