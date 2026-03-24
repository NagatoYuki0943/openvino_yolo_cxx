// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#ifndef OPENVINO_YOLO11_DET_INFER_HPP
#define OPENVINO_YOLO11_DET_INFER_HPP
#pragma once

#include <string>
#include <vector>
#include <array>
#include <random>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

namespace yolo
{

    struct YoloDetectResult
    {
        int class_id;
        std::string class_name;
        float confidence;
        int left;
        int top;
        int right;
        int bottom;
        int track_id = -1;
    };

    class OpenvinoYolo11DetInference
    {
    private:
        ov::Core _core;
        ov::CompiledModel _compiled_model;   // OpenVINO compiled model
        ov::InferRequest _inference_request; // OpenVINO inference request

        cv::Size _model_input_shape;  // Input shape of the model (width, height)
        cv::Size _model_output_shape; // Output shape of the model (width, height)

    public:
        std::vector<std::string> _classes{
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"};
        void init_model(
            const std::string &model_path,
            const cv::Size &model_input_shape = cv::Size(640, 640))
        {
            // std::cout << "\n---------- start init_model ----------" << std::endl;

            ov::Core core;                                                  // OpenVINO core object
            std::shared_ptr<ov::Model> model = core.read_model(model_path); // Read the model from file

            // If the model has dynamic shapes, reshape it to the specified input shape
            if (model->is_dynamic())
                model->reshape({1, 3, static_cast<long int>(model_input_shape.height), static_cast<long int>(model_input_shape.width)});

            int width, height;

            // Get input shape from the model
            const std::vector<ov::Output<ov::Node>> inputs = model->inputs();
            const ov::Shape input_shape = inputs[0].get_shape();
            // std::cout << "output_shape: " << input_shape << std::endl;
            height = input_shape[2];
            width = input_shape[3];
            this->_model_input_shape = cv::Size(width, height);
            // std::cout << "_model_input_shape shape(wxh): " << this->_model_input_shape << std::endl;

            // Get output shape from the model
            const std::vector<ov::Output<ov::Node>> outputs = model->outputs();
            const ov::Shape output_shape = outputs[0].get_shape();
            // std::cout << "output_shape: " << output_shape << std::endl;
            height = output_shape[1];
            width = output_shape[2];
            this->_model_output_shape = cv::Size(width, height);
            // std::cout << "_model_output_shape shape(wxh): " << this->_model_output_shape << std::endl;

            // pre_process setup for the model
            ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
            ppp.input()
                .tensor()
                .set_element_type(ov::element::u8)
                .set_layout("NHWC")
                .set_color_format(ov::preprocess::ColorFormat::BGR);
            ppp.input()
                .preprocess()
                .convert_color(ov::preprocess::ColorFormat::RGB)
                // .convert_layout("NCHW")
                .convert_element_type(ov::element::f32)
                .scale({255, 255, 255});
            ppp.input()
                .model()
                .set_layout("NCHW");
            ppp.output()
                .tensor()
                .set_element_type(ov::element::f32);
            model = ppp.build(); // Build the preprocessed model

            // Compile the model for inference
            this->_compiled_model = core.compile_model(model, "AUTO");
            this->_inference_request = this->_compiled_model.create_infer_request(); // Create inference request

            // std::cout << "---------- init_model end ----------\n"
            //           << std::endl;
        }

        // Constructor to initialize the model with specified input shape
        OpenvinoYolo11DetInference(const std::string &model_path, const cv::Size model_input_shape)
        {
            init_model(model_path, model_input_shape);
        }

        std::vector<YoloDetectResult> predict(
            cv::Mat &image,
            const float confidence_threshold = 0.25,
            const float NMS_threshold = 0.5)
        {
            // std::cout << "\n---------- start predict ----------" << std::endl;

            float scale_factor = pre_process(image); // Preprocess the input image
            this->_inference_request.infer();          // Run inference
            auto detect_results = post_process(
                confidence_threshold,
                NMS_threshold,
                scale_factor,
                image.size()); // Postprocess the inference results

            // std::cout << "---------- predict end ----------\n"
            //           << std::endl;
            return detect_results;
        }

        // Method to preprocess the input image
        float pre_process(const cv::Mat &image)
        {
            // std::cout << "\n---------- start pre_process ----------" << std::endl;

            float scale_factor = std::min(
                static_cast<float>(this->_model_input_shape.width) / static_cast<float>(image.cols),
                static_cast<float>(this->_model_input_shape.height) / static_cast<float>(image.rows));
            int new_width = static_cast<int>(image.cols * scale_factor);
            int new_height = static_cast<int>(image.rows * scale_factor);

            cv::Mat resized_image;
            // 缩放高宽的长边为最大长度
            cv::resize(image, resized_image, {new_width, new_height}, 0, 0, cv::INTER_AREA); // Resize the image to imagech the model input shape

            // std::cout << "resized_image size(wxh): " << resized_image.size() << std::endl;
            // cv::imshow("resized_image", resized_image);
            // cv::waitKey(0);

            // 填充bottom和right的长度
            int delta_w = this->_model_input_shape.width - new_width;
            int delta_h = this->_model_input_shape.height - new_height;
            cv::copyMakeBorder(
                resized_image,
                resized_image,
                0,
                delta_h,
                0,
                delta_w,
                cv::BORDER_CONSTANT,
                cv::Scalar(0, 0, 0));

            // std::cout << "copyMakeBorder image size(wxh): " << resized_image.size() << std::endl;
            // cv::imshow("resized_image", resized_image);
            // cv::waitKey(0);

            ov::Tensor input_tensor = this->_inference_request.get_input_tensor();
            uint8_t *input_data = input_tensor.data<uint8_t>();
            size_t bytes_to_copy = resized_image.total() * resized_image.elemSize();
            memcpy(input_data, resized_image.data, bytes_to_copy);

            this->_inference_request.set_input_tensor(input_tensor); // Set input tensor for inference

            // std::cout << "---------- pre_process end ----------\n"
            //           << std::endl;
            return scale_factor;
        }

        // Method to postprocess the inference results
        std::vector<YoloDetectResult> post_process(
            const float confidence_threshold,
            const float NMS_threshold,
            const float scale_factor,
            const cv::Size &original_shape)
        {
            // std::cout << "\n---------- start post_process ----------" << std::endl;

            std::vector<int> class_list;
            std::vector<float> confidence_list;
            std::vector<cv::Rect> box_list;
            std::vector<cv::Rect> nms_box_list; // 专门用于 NMS 的带偏移量框列表

            // 1. 获取原始的 84 x 8400 矩阵
            // 8400 列：代表模型预测出的 8400 个候选框（Anchors）。
            // 84 行：代表每个候选框的 84 个属性。前 4 个是中心点坐标和宽高 (cx,cy,w,h)，后 80 个是该框属于 80 个类别的得分。
            const float *detections = this->_inference_request.get_output_tensor().data<const float>();
            // Create OpenCV imagerix from output tensor
            // clone 是因为 cv::Mat 使用指针创建 Mat 时没有拷贝数据, 所以这里需要 clone 一份数据
            const cv::Mat detection_outputs = cv::Mat(this->_model_output_shape, CV_32F, (float *)detections).clone();
            // std::cout << "detection_outputs shape(wxh): " << detection_outputs.size() << std::endl;

            // 设定一个足够大的常量作为偏移基数 (通常 YOLO 输入是 640 或 1280，4096 绝对够用)
            const int max_wh = 4096;

            // 2. 将其转置为 8400 x 84 (行数变成 8400，列数变成 84)
            cv::Mat transposed_outputs;
            cv::transpose(detection_outputs, transposed_outputs);
            // std::cout << "transposed_outputs shape(wxh): " << transposed_outputs.size() << std::endl;

            // Iterate over detections and collect class IDs, confidence scores, and bounding boxes
            // 循环 8400 次，每次处理一列（即一个预测框）
            for (int i = 0; i < transposed_outputs.rows; ++i)
            {
                // transposed_outputs.row(i)：取第 i 个预测框的所有数据（84个值）。
                // .colRange(4, transposed_outputs.cols)：跳过前 4 个位置坐标参数，截取第 4 到第 83 行。这 80 个值就是该框在 80 个分类上的概率得分。
                const cv::Mat _classesscores = transposed_outputs.row(i).colRange(4, transposed_outputs.cols);

                // 找出得分最高的值赋给 score，并把该得分的**索引（类别 ID）**赋给 class_id.y
                cv::Point class_id;
                double score;
                cv::minMaxLoc(_classesscores, nullptr, &score, nullptr, &class_id); // Find the class with the highest score

                // Check if the detection meets the confidence threshold
                if (score > confidence_threshold)
                {
                    class_list.push_back(class_id.x);
                    confidence_list.push_back(score);

                    const float cx = transposed_outputs.at<float>(i, 0);
                    const float cy = transposed_outputs.at<float>(i, 1);
                    const float w = transposed_outputs.at<float>(i, 2);
                    const float h = transposed_outputs.at<float>(i, 3);

                    // 1. 保存原始真实的预测框（用于最终结果提取和绘制）
                    cv::Rect box;
                    box.x = static_cast<int>((cx - w / 2));
                    box.y = static_cast<int>((cy - h / 2));
                    box.width = static_cast<int>(w);
                    box.height = static_cast<int>(h);
                    box_list.push_back(box);

                    // 2. 生成带偏移量的框（仅用于送入 NMS 函数）
                    cv::Rect nms_box;
                    nms_box.x = box.x + class_id.y * max_wh;
                    nms_box.y = box.y + class_id.y * max_wh;
                    nms_box.width = box.width;
                    nms_box.height = box.height;
                    nms_box_list.push_back(nms_box);
                }
            }

            // Apply Non-Maximum Suppression (NMS) to filter overlapping bounding boxes
            std::vector<int> NMS_result;
            // 【核心修改】这里传入的是带偏移量的 nms_box_list，而不是原来的 box_list
            cv::dnn::NMSBoxes(nms_box_list, confidence_list, confidence_threshold, NMS_threshold, NMS_result);

            std::vector<YoloDetectResult> results;
            // Collect final detections after NMS
            for (int i = 0; i < NMS_result.size(); ++i)
            {
                YoloDetectResult result;
                int id = NMS_result[i];

                result.class_id = class_list[id];
                result.class_name = this->_classes[result.class_id];
                result.confidence = confidence_list[id];

                // 这里依然使用未加偏移量的真实 box_list[id] 来还原坐标
                auto box = get_bounding_box(box_list[id], scale_factor, original_shape);
                result.left = box[0];
                result.top = box[1];
                result.right = box[2];
                result.bottom = box[3];

                results.push_back(result);
            }

            // std::cout << "---------- post_process end ----------\n"
            //           << std::endl;
            return results;
        }

        // Method to get the bounding box in the correct scale
        std::array<int, 4> get_bounding_box(const cv::Rect &src, const float scale_factor, const cv::Size &original_shape)
        {
            // 1. 将基于 640x640 尺度的坐标和宽高，按比例还原回原图尺度
            int left = static_cast<int>(src.x / scale_factor);
            int top = static_cast<int>(src.y / scale_factor);
            int width = static_cast<int>(src.width / scale_factor);
            int height = static_cast<int>(src.height / scale_factor);

            // 2. 构建还原后的预测框
            cv::Rect scaled_box(left, top, width, height);

            // 3. 构建代表原始图像物理边界的框
            cv::Rect image_rect(0, 0, original_shape.width, original_shape.height);

            // 4. 【核心安全机制】求交集，自动切除越界部分
            cv::Rect safe_box = scaled_box & image_rect;

            // 5. 提取安全截断后的左上角和右下角坐标
            int safe_left = safe_box.x;
            int safe_top = safe_box.y;
            int safe_right = safe_box.x + safe_box.width;
            int safe_bottom = safe_box.y + safe_box.height;

            return {safe_left, safe_top, safe_right, safe_bottom};
        }
    };

    // 辅助函数：根据固定的 ID 生成稳定且明亮的颜色
    cv::Scalar GetColorForId(int id)
    {
        // 使用质数进行简单的哈希计算，放大不同 ID 之间的颜色差异
        // 取模 136 然后加上 120，确保颜色通道值在 120-255 之间（保证明亮度）
        int r = 120 + ((id * 37) % 136);
        int g = 120 + ((id * 73) % 136);
        int b = 120 + ((id * 109) % 136);
        return cv::Scalar(b, g, r);
    }

    cv::Mat draw_detected_object(cv::Mat &image, const std::vector<YoloDetectResult> &detect_results)
    {
        for (const auto &result : detect_results)
        {
            const float &confidence = result.confidence;
            const std::string &class_name = result.class_name;
            const int &class_id = result.class_id;

            // 优先使用 track_id 获取颜色。如果未使用追踪器，则退化为按类别区分颜色
            int id_to_color = (result.track_id >= 0) ? result.track_id : class_id;
            const cv::Scalar color = GetColorForId(id_to_color);

            // 绘制目标边界框
            cv::rectangle(image, cv::Point(result.left, result.top), cv::Point(result.right, result.bottom), color, 2);

            // 准备标签文本
            std::string classString = class_name + " " + std::to_string(confidence).substr(0, 4);
            if (result.track_id >= 0)
            {
                classString += " ID:" + std::to_string(result.track_id);
            }

            // 计算文本框大小
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 0.75, 1, 0);

            // 动态计算文本框和文字的 Y 坐标，防止物体在图像最顶端时文字出界
            int box_top, text_top;
            if (result.top > textSize.height + 5)
            {
                box_top = result.top - textSize.height - 10;
                text_top = result.top - 5;
            }
            else
            {
                // 如果物体太靠上，把文字框画在边界框内部
                box_top = result.top;
                text_top = result.top + textSize.height + 5;
            }

            // 绘制文本背景框 (FILLED)
            cv::rectangle(image, {result.left, box_top}, {result.left + textSize.width, box_top + textSize.height + 10}, color, cv::FILLED);

            // 绘制文本（使用黑色字体保证在明亮背景上的对比度）
            cv::putText(image, classString, {result.left, text_top}, cv::FONT_HERSHEY_DUPLEX, 0.75, cv::Scalar(0, 0, 0), 1, 0);
        }
        return image;
    }

} // namespace yolo

#endif // OPENVINO_YOLO11_DET_INFER_HPP
