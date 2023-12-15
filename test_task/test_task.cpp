// test_task.cpp : Defines the entry point for the application.
//

#include "test_task.h"

using namespace std;
using namespace cv;

void FastDetector(string image_path) {
    Ptr<FeatureDetector> fastDetector = FastFeatureDetector::create();

    // Завантаження зображення в сірих відтінках
    Mat image = imread(image_path, IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Could not open or find the image!" << endl;
    }

    // Знаходження ключових точок
    vector<KeyPoint> keypoints;
    fastDetector->detect(image, keypoints);

    // Виведення результатів
    Mat output;
    Scalar color(0, 0, 255);
    drawKeypoints(image, keypoints, output, color);
    imshow("Fast Keypoints", output);
    waitKey(0);

    cout << "Total Keypoints in Fast Detector: " << keypoints.size() << endl;
    string filename = "new_" + image_path;
    imwrite(filename, image);
}


void KLTDetector(string gif_path) {
    //Відкрити відео файл
    VideoCapture cap(gif_path);

    // Перевірка чи відео успішно відкрито
    if (!cap.isOpened()) {
        cerr << "Error: Could not open video file." << endl;
    }

    // Отримання першого кадру
    Mat prevFrame;
    cap >> prevFrame;

    // Перевірка чи вдалося отримати кадр
    if (prevFrame.empty()) {
        cerr << "Error: Could not read video frame." << endl;
    }

    // Конвертація у відтінки сірого
    cvtColor(prevFrame, prevFrame, COLOR_BGR2GRAY);

    // Визначення кутів за допомогою Shi-Tomasi detector
    vector<Point2f> prevCorners;
    goodFeaturesToTrack(prevFrame, prevCorners, 100, 0.01, 10);




    // Параметри алгоритму KLT
    TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
    Size winSize(31, 31);

    // Основний цикл обробки кадрів
    while (true) {
        // Отримання наступного кадру
        Mat nextFrame;
        cap >> nextFrame;

        if (nextFrame.empty()) {
            cerr << "Error: Could not read next video frame." << endl;
            break;
        }


        // Конвертація у відтінки сірого
        cvtColor(nextFrame, nextFrame, COLOR_BGR2GRAY);

        // Відстеження кутів за допомогою алгоритму KLT
        vector<Point2f> nextCorners;
        vector<uchar> status;
        vector<float> err;
        calcOpticalFlowPyrLK(prevFrame, nextFrame, prevCorners, nextCorners, status, err, winSize, 3, termcrit, 0, 0.001);

        // Відображення результатів на кадрі
        for (size_t i = 0; i < prevCorners.size(); ++i) {
            if (status[i] == 1) {
                circle(nextFrame, nextCorners[i], 3, Scalar(0, 255, 0), -1, 8);
                line(nextFrame, prevCorners[i], nextCorners[i], Scalar(0, 0, 255), 1, 8, 0);
            }
        }

        // Відображення кадру
        imshow("KLT Tracking", nextFrame);

        // Збереження кадрів для наступної ітерації
        prevFrame = nextFrame.clone();
        prevCorners = nextCorners;

        // Очікування натискання клавіші для виходу
        if (waitKey(30) == 27) {
            break;  // Клавіша 'Esc' для виходу
        }
    }

    // Завершення
    cap.release();
    destroyAllWindows();
}

int main()
{
    //Вивід FastDetector
    string images[] = { "signal-2023-12-14-212155_002.jpeg", "signal-2023-12-14-212155_003.jpeg" };
    for (int i = 0; i < 2; i++) {
        FastDetector(images[i]);
    }

    //Вивід KLTDetector

    KLTDetector("4.mp4");




    return 0;
}