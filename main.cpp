#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include "yolo/openvino_yolo11_det_inference.hpp"
#include "ByteTrack/BYTETracker.h"
#include "utils/functions.hpp"
#include "utils/point_polygon_test.hpp"
#include "global_vars.hpp"
#include "global_funcs.hpp"

namespace fs = std::filesystem;

int predict_image(const Global::GereralConfig &config, const std::string &image_path, bool filter_boxes_in_polygon = false)
{
    std::string output_path = fs::path(image_path).stem().string() + "--predict.jpg";
    std::cout << "save predict image to " << output_path << std::endl;

    cv::Mat image = cv::imread(image_path);
    if (image.empty())
    {
        std::cout << "image_path: " << image_path << "read failed" << std::endl;
        return -1;
    }
    std::cout << "image size: " << image.size() << std::endl;
    // cv::imshow("image", image);
    // cv::waitKey(0);

    // Initialize the YOLO inference with the specified model and parameters
    yolo::OpenvinoYolo11DetInference inference = {
        config.detect_config.model_path,
        config.detect_config.model_input_shape,
        config.detect_config.classes};

    // Run inference on the input image
    auto detect_boxes = inference.infer(image, config.detect_config.conf_threshold, config.detect_config.nms_threshold);
    std::cout << "detect_boxes num = " << detect_boxes.size() << std::endl;
    std::cout << "detect_boxes:" << std::endl;
    for (const auto &detect_box : detect_boxes)
    {
        std::cout << "    class_id: " << detect_box.class_id
                  << ", class_name: " << detect_box.class_name
                  << ", confidence: " << detect_box.confidence
                  << ", box : [" << detect_box.left
                  << ", " << detect_box.top
                  << ", " << detect_box.right
                  << ", " << detect_box.bottom << "]" << std::endl;
    }

    cv::Mat draw_image = image.clone();

    // 过滤出在多边形内部的目标
    if (filter_boxes_in_polygon)
    {
        // 创建一个多边形,每个点都是 (x, y), 左上角是原点
        std::vector<cv::Point> polygon = {
            cv::Point(0, 200),
            cv::Point(600, 700),
            cv::Point(50, 1000),
            cv::Point(250, 800),
            cv::Point(100, 700)};
        auto inside_boxes = detect_utils::filter_boxes_in_polygon(detect_boxes, polygon);
        std::cout << "inside_boxes num = " << inside_boxes.size() << std::endl;

        // 绘制多边形
        detect_utils::draw_closed_polygon(draw_image, polygon);
        detect_boxes = std::move(inside_boxes);
    }

    detect_utils::draw_detected_object(draw_image, detect_boxes);

    cv::imwrite(output_path, draw_image);
    std::cout << "Image processing complete. Output saved to: " << output_path << std::endl;

    // Display the image with the detections
    cv::imshow("draw_image", draw_image);
    cv::waitKey(0);

    return 0;
}

int predict_video(const Global::GereralConfig &config, const std::string &video_path)
{
    std::string output_path = fs::path(video_path).stem().string() + "--predict.mp4";
    std::cout << "save predict video to " << output_path << std::endl;

    // 1. 初始化 YOLO 推理
    yolo::OpenvinoYolo11DetInference inference = {
        config.detect_config.model_path,
        config.detect_config.model_input_shape,
        config.detect_config.classes};

    // 2. 打开输入视频
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        std::cout << "Error: Could not open video file: " << video_path << std::endl;
        return -1;
    }

    // 获取视频的基本属性，用于初始化 VideoWriter
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    // 3. 初始化 VideoWriter (写入视频文件)
    // 这里使用 mp4v 编码器生成 .mp4 格式的视频
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frame_width, frame_height));
    if (!writer.isOpened())
    {
        std::cout << "Error: Could not open VideoWriter for: " << output_path << std::endl;
        return -1;
    }

    cv::Mat frame;
    int frame_count = 0;
    std::cout << "Start processing video..." << std::endl;

    // 4. 循环读取视频帧
    while (cap.read(frame))
    {
        if (frame.empty())
        {
            break; // 视频结束
        }

        // 进行目标检测
        auto detect_boxes = inference.infer(frame, config.detect_config.conf_threshold, config.detect_config.nms_threshold);
        std::cout << "detect_boxes size: " << detect_boxes.size() << std::endl;

        // 将检测框绘制到当前帧上
        detect_utils::draw_detected_object(frame, detect_boxes);

        // 将处理后的帧写入输出视频文件
        writer.write(frame);

        // 实时显示处理过程 (可选)
        cv::imshow("Video Inference", frame);
        // 按下 ESC 键 (ASCII: 27) 可以提前终止处理，等待时间为1ms
        if (cv::waitKey(1) == 27)
        {
            std::cout << "Video processing interrupted by user." << std::endl;
            break;
        }

        frame_count++;
        if (frame_count % 30 == 0)
        {
            std::cout << "Processed " << frame_count << " frames..." << std::endl;
        }
    }

    // 5. 释放资源
    cap.release();
    writer.release();
    cv::destroyAllWindows();

    std::cout << "Video processing complete. Output saved to: " << output_path << std::endl;
    return 0;
}

int track_video(const Global::GereralConfig &config, const std::string &video_path, bool enable_multi_class_tracking = true)
{
    std::string output_path = fs::path(video_path).stem().string() + "--track" + (enable_multi_class_tracking ? "--in_multi_class" : "--in_single_class") + ".mp4";
    std::cout << "save track video to " << output_path << std::endl;

    // 1. 初始化 YOLO 推理
    yolo::OpenvinoYolo11DetInference inference = {
        config.detect_config.model_path,
        config.detect_config.model_input_shape,
        config.detect_config.classes};

    // BYTETracker 参数解释
    // max_time_lost 死亡倒计时，目标丢失（未匹配到检测框）后，在内存中保留等待重新出现的总帧数
    // track_high_thresh 高分界线，得分大于此值的框为“高分框”，参与第一轮常规匹配。
    // track_low_thresh 低分界线，得分在此值与高分界线之间的为“低分框”，参与第二轮遮挡修补；低于此值的框直接丢弃。
    // new_track_thresh 出生门槛，只有得分大于此值的检测框，才能被初始化为全新的追踪目标。
    // match_thresh 认亲标准，判定检测框与已有轨迹“是否为同一目标”的匹配代价容忍度（通常基于 IoU）。越高，越容易匹配；越低，越难匹配。
    //  什么时候应该调高（比如 0.8 ~ 0.9）？
    //      视频帧率（FPS）较低时。
    //      目标运动速度极快时。
    //      原因： 这种情况下，目标在相邻两帧之间的位移跨度很大，导致上一帧的预测框和这一帧的检测框交集 (IoU) 很小。如果不把容忍度调高，追踪器会认为老目标消失了，新目标出现了，导致疯狂闪烁和切换 ID。
    //  什么时候应该调低（比如 0.4 ~ 0.5）？
    //      画面极其拥挤密集时（例如密集的车流、十字路口的人群）。
    //      原因： 当多个目标靠得非常近时，如果追踪器太宽容，很容易发生 ID 劫持（比如 A 车和 B 车并排，追踪器把 A 的 ID 错误地连到了 B 的检测框上）。降低阈值能逼迫追踪器变得“严谨”，只认准那个跟历史轨迹重合度最高的目标。
    // min_hits 连续追踪到多少帧才输出

    // 2. 初始化追踪器
    std::map<int, ByteTrack::BYTETracker> trackers;
    if (!enable_multi_class_tracking)
    {
        // 单个追踪器
        trackers[0] = ByteTrack::BYTETracker{
            config.track_config.max_time_lost,
            config.track_config.track_high_thresh,
            config.track_config.track_low_thresh,
            config.track_config.new_track_thresh,
            config.track_config.match_thresh,
            config.track_config.min_hits};
    }
    else
    {
        // 每个类别1个追踪器
        for (const auto &[id, name] : config.detect_config.classes)
        {
            trackers[id] = ByteTrack::BYTETracker{
                config.track_config.max_time_lost,
                config.track_config.track_high_thresh,
                config.track_config.track_low_thresh,
                config.track_config.new_track_thresh,
                config.track_config.match_thresh,
                config.track_config.min_hits};
        }
    }
    std::cout << "trackers size: " << trackers.size() << std::endl;

    // 3. 打开输入视频
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        std::cout << "Error: Could not open video file: " << video_path << std::endl;
        return -1;
    }

    // 获取视频的基本属性，用于初始化 VideoWriter
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    // 4. 初始化 VideoWriter (写入视频文件)
    // 这里使用 mp4v 编码器生成 .mp4 格式的视频
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frame_width, frame_height));
    if (!writer.isOpened())
    {
        std::cout << "Error: Could not open VideoWriter for: " << output_path << std::endl;
        return -1;
    }

    cv::Mat frame;
    int frame_count = 0;
    std::cout << "Start processing video..." << std::endl;

    // 4. 循环读取视频帧
    while (cap.read(frame))
    {
        if (frame.empty())
        {
            break; // 视频结束
        }

        // 进行目标检测
        auto detect_boxes = inference.infer(frame, config.detect_config.conf_threshold, config.detect_config.nms_threshold);
        std::cout << "detect_boxes size: " << detect_boxes.size() << std::endl;

        // 追踪
        std::vector<Global::YoloDetectBox> track_boxes;
        track_boxes.reserve(detect_boxes.size());
        if (!enable_multi_class_tracking)
        {
            auto &tracker = trackers[0];
            std::vector<ByteTrack::Object> track_objects = {};
            std::vector<ByteTrack::STrack> tracklets = {};
            std::vector<ByteTrack::STrack> lostTracklets = {};
            std::vector<ByteTrack::STrack> removedTracklets = {};
            int index = 0;
            for (const auto &detect_box : detect_boxes)
            {
                ByteTrack::Object obj;
                obj.target_id = index;
                obj.class_id = detect_box.class_id;
                obj.prob = detect_box.confidence;
                obj.rect = cv::Rect_<float>(
                    static_cast<float>(detect_box.left),
                    static_cast<float>(detect_box.top),
                    static_cast<float>(detect_box.right - detect_box.left),
                    static_cast<float>(detect_box.bottom - detect_box.top));
                track_objects.push_back(obj);
                index += 1;
            }

            // Tracking
            tracker.update(track_objects, tracklets, lostTracklets, removedTracklets);
            std::cout << "tracklets size: " << tracklets.size() << std::endl;
            std::cout << "lostTracklets size: " << lostTracklets.size() << std::endl;
            std::cout << "removedTracklets size: " << removedTracklets.size() << std::endl;

            for (const auto &tracklet : tracklets)
            {
                // 创建新数据
                Global::YoloDetectBox box;
                box.track_id = tracklet.track_id;
                box.class_id = tracklet.class_id;
                box.class_name = inference._classes[box.class_id];
                box.confidence = tracklet.score;
                box.left = tracklet.tlwh[0];
                box.top = tracklet.tlwh[1];
                box.right = tracklet.tlwh[0] + tracklet.tlwh[2];
                box.bottom = tracklet.tlwh[1] + tracklet.tlwh[3];

                // 使用原始检测框, 只是添加 track_id
                // auto box = detect_boxes[tracklet.target_id];
                // box.track_id = tracklet.track_id;

                track_boxes.push_back(box);
            }
        }
        else
        {
            // 将检测结果按照类别进行分类
            auto class_map = detect_utils::classify_boxed_by_class(detect_boxes);

            for (const auto &[class_id, detect_indexes] : class_map)
            {
                std::cout << "class_id: " << class_id << " detect_indexes: [";
                for (int index : detect_indexes)
                {
                    std::cout << index << ", ";
                }
                std::cout << "]" << std::endl;

                if (trackers.find(class_id) == trackers.end())
                {
                    std::cout << "Warning: Unknown class_id " << class_id << " detected!" << std::endl;
                    continue; // 跳过未配置的类别
                }
                auto &tracker = trackers.at(class_id);

                std::vector<ByteTrack::Object> track_objects = {};
                std::vector<ByteTrack::STrack> tracklets = {};
                std::vector<ByteTrack::STrack> lostTracklets = {};
                std::vector<ByteTrack::STrack> removedTracklets = {};

                // 转换格式
                for (const auto &index : detect_indexes)
                {
                    auto detect_box = detect_boxes[index];

                    ByteTrack::Object obj;
                    obj.target_id = index;
                    obj.class_id = detect_box.class_id;
                    obj.prob = detect_box.confidence;
                    obj.rect = cv::Rect_<float>(
                        static_cast<float>(detect_box.left),
                        static_cast<float>(detect_box.top),
                        static_cast<float>(detect_box.right - detect_box.left),
                        static_cast<float>(detect_box.bottom - detect_box.top));
                    track_objects.push_back(obj);
                }

                // Tracking
                tracker.update(track_objects, tracklets, lostTracklets, removedTracklets);
                std::cout << "class_id: " << class_id << " tracklets size: " << tracklets.size() << std::endl;
                std::cout << "class_id: " << class_id << " lostTracklets size: " << lostTracklets.size() << std::endl;
                std::cout << "class_id: " << class_id << " removedTracklets size: " << removedTracklets.size() << std::endl;

                // 转换格式
                for (const auto &tracklet : tracklets)
                {
                    // 创建新数据
                    Global::YoloDetectBox box;
                    box.track_id = tracklet.track_id;
                    box.class_id = tracklet.class_id;
                    box.class_name = inference._classes[box.class_id];
                    box.confidence = tracklet.score;
                    box.left = tracklet.tlwh[0];
                    box.top = tracklet.tlwh[1];
                    box.right = tracklet.tlwh[0] + tracklet.tlwh[2];
                    box.bottom = tracklet.tlwh[1] + tracklet.tlwh[3];

                    // 使用原始检测框, 只是添加 track_id
                    // auto box = detect_boxes[tracklet.target_id];
                    // box.track_id = tracklet.track_id;

                    track_boxes.push_back(box);
                }
            }
            std::cout << std::endl;
        }

        // 将检测框绘制到当前帧上
        detect_utils::draw_detected_object(frame, track_boxes);

        // 将处理后的帧写入输出视频文件
        writer.write(frame);

        // 实时显示处理过程 (可选)
        cv::imshow("Video Inference", frame);
        // 按下 ESC 键 (ASCII: 27) 可以提前终止处理，等待时间为1ms
        if (cv::waitKey(1) == 27)
        {
            std::cout << "Video processing interrupted by user." << std::endl;
            break;
        }

        frame_count++;
        if (frame_count % 30 == 0)
        {
            std::cout << "Processed " << frame_count << " frames..." << std::endl;
        }

        std::cout << std::endl;
    }

    // 5. 释放资源
    cap.release();
    writer.release();
    cv::destroyAllWindows();

    std::cout << "Video processing complete. Output saved to: " << output_path << std::endl;
    return 0;
}

int main(int argc, char *argv[])
{
    std::cout << "============================================================" << std::endl;
    std::cout << "OpenVINO YOLO C++ Demo help: " << std::endl;
    std::cout << "    for predict image, usage: " << argv[0] << " predict_image <model_config_path> <image_path>" << std::endl;
    std::cout << "    for predict video, usage: " << argv[0] << " predict_video <model_config_path> <video_path>" << std::endl;
    std::cout << "    for track video, usage: " << argv[0] << " track_video <model_config_path> <video_path> <0 or 1:enable_multi_class_tracking>" << std::endl;
    std::cout << "    for filter boxes in polygon(default box), usage: " << argv[0] << " filter_boxes <model_config_path> <image_path>" << std::endl;
    std::cout << "============================================================" << std::endl;

    // std::cout << "argc: " << argc << std::endl;
    // std::cout << "program name: " << argv[0] << std::endl;
    std::vector<std::string> args(argv, argv + argc);

    if (args.size() < 4)
    {
        std::cout << "Error: Insufficient arguments provided." << std::endl;
        return -1;
    }

    std::string mode = args[1];
    std::string config_path = args[2];
    std::string image_path = args[3];

    std::cout << "mode: " << mode << std::endl;
    std::cout << "config_path: " << config_path << std::endl;
    std::cout << "image_path: " << image_path << std::endl;

    if (!fs::exists(config_path))
    {
        std::cout << "config_path: " << config_path << " not exist" << std::endl;
        return -1;
    }
    auto config = Global::read_config(config_path);

    if (!fs::exists(image_path))
    {
        std::cout << "image_path/video_path: " << image_path << " not exist" << std::endl;
        return -1;
    }

    int res = 0;
    if (mode == "predict_image")
    {
        res = predict_image(config, image_path);
    }
    else if (mode == "predict_video")
    {
        res = predict_video(config, image_path);
    }
    else if (mode == "track_video")
    {
        bool enable_multi_class_tracking = true;
        if (args.size() > 4)
        {
            enable_multi_class_tracking = std::stoi(args[4]);
        }

        res = track_video(config, image_path, enable_multi_class_tracking);
    }
    else if (mode == "filter_boxes")
    {
        detect_utils::test_filter_boxes_in_polygon();
        res = predict_image(config, image_path, true);
    }
    else
    {
        res = -1;
        std::cout << "Error: Invalid mode provided." << std::endl;
    }

    return res;
}
