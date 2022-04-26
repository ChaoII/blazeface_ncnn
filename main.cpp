//
// Created by aichao on 2022/4/26.
//
#include "blazeface.h"

static int draw_fps(cv::Mat &rgb) {
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f) {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--) {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f) {
            return 0;
        }

        for (int i = 0; i < 10; i++) {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}

int main() {


    BlazeFace blazeFace;
    blazeFace.load(320, false);
    cv::VideoCapture cap(1);
    cv::Mat frame;
    while (true) {
        double st = ncnn::get_current_time();
        cap >> frame;
        std::vector<FaceObject> face_objects;
        blazeFace.detect(frame, face_objects);
        blazeFace.draw(frame, face_objects);
        draw_fps(frame);
        std::cout << (ncnn::get_current_time() - st) << std::endl;
        cv::imshow("12", frame);
        if (cv::waitKey(30) == char(27)) {
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();

    return 0;
}