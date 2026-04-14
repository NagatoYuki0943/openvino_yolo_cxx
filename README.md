# openvino yolo 目标检测

参考 https://github.com/ultralytics/ultralytics/tree/main/examples/YOLOv8-OpenVINO-CPP-Inference

# 依赖

- opencv
- openvino
- eigen3 (目标追踪)

> 目标追踪使用了 [junhui-ng/ByteTrack-CPP: C++ implementation of ByteTrack algorithm](https://github.com/junhui-ng/ByteTrack-CPP) 项目源码

# 配置

需要配置 `CMakePresets.json` 中的依赖库的 cmake 路径

```json
            "cacheVariables": {
                "OpenVINO_DIR": "C:/cxx/OpenVINO/runtime/cmake",
                "OpenCV_DIR": "C:/cxx/opencv/build",
                "Eigen3_DIR": "C:/cxx/eigen3_install/share/eigen3/cmake",
                "CMAKE_PREFIX_PATH": "C:/miniconda3/envs/cxx/Library",
            },
```

# 编译好的程序使用方式

例子

```powershell
============================================================
OpenVINO YOLO C++ Demo help:
    for predict image, usage: .\main.exe predict_image <model_config_path> <image_path>
    for predict video, usage: .\main.exe predict_video <model_config_path> <video_path>
    for track video, usage: .\main.exetrack_video <model_config_path> <video_path> <0 or 1:enable_multi_class_tracking>
    for filter boxes by polygon(default box), usage: .\main.exe filter_boxes <model_config_path> <image_path>
============================================================

# for predict image
# 执行后会在执行目录生成 bus--predict.jpg
.\main.exe predict_image ..\..\..\models\metadata.json ..\..\..\images\bus.jpg

# for predict video
# 执行后会在执行目录生成 MOT16-08-raw--predict.mp4
.\main.exe predict_video ..\..\..\models\metadata.json ..\..\..\videos\MOT16-08-raw.mp4

# for track video
# 执行后会在执行目录生成 MOT16-08-raw--track--in_multi_class.mp4
.\main.exe track_video ..\..\..\models\metadata.json ..\..\..\videos\MOT16-08-raw.mp4 1
# 执行后会在执行目录生成 MOT16-08-raw--track--in_single_class.mp4
.\main.exe track_video ..\..\..\models\metadata.json ..\..\..\videos\MOT16-08-raw.mp4 0

# for filter boxes
# 执行后会在执行目录生成 test_filter_boxes_in_polygon.jpg, bus--predict.jpg
.\main.exe filter_boxes ..\..\..\models\metadata.json ..\..\..\images\bus.jpg
```
