package mitpc.medsoft.component.service;

import io.minio.messages.Item;
import jakarta.annotation.Resource;
import lombok.Getter;
import lombok.extern.log4j.Log4j;
import mitpc.medsoft.minio.Minio;
import mitpc.medsoft.model.Face;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_java;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import org.opencv.core.MatOfInt;
import org.opencv.face.LBPHFaceRecognizer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

@Log4j
@Configuration
public class FaceCascadeConfig {
    @Resource
    private Minio minio;

    @Getter
    private List<String> picNames = new ArrayList<>();

    @Bean
    public Face faceCascade() {
        Loader.load(opencv_java.class);

        File xml = minio.getFile("/src/files/faceRecognition/haarcascade_frontalface_default.xml").getData();
        log.info("Cascade Classifier XML Path: " + xml.getPath());
        CascadeClassifier cascade = new CascadeClassifier(xml.getPath());
        if (cascade.empty()) {
            throw new IllegalStateException("Failed to load CascadeClassifier");
        }

        LBPHFaceRecognizer faceRecognizer = LBPHFaceRecognizer.create();

        List<Item> itemList = minio.getFiles("/faceRecognition").getData();
        List<String> fileNames = new ArrayList<>();
        for (Item item : itemList) {
            fileNames.add(item.objectName());
        }
        List<File> fileList = new ArrayList<>();
        for (String fileName : fileNames) {
            fileList.add(minio.getFile(fileName).getData());
        }

        List<org.opencv.core.Mat> images = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        int labelCounter = 1;
        for (File file : fileList) {
            log.info("Loading image: " + file.getAbsolutePath());
            Mat image = loadImage(file);

            if (image == null || image.empty()) {
                log.warn("Failed to load image or image is empty: " + file.getAbsolutePath());
                continue;
            }

            org.opencv.core.Mat convertedMat = convertMat(image);
            if (convertedMat.empty()) {
                log.warn("Converted Mat is empty for image: " + file.getAbsolutePath());
                continue;
            }

            org.opencv.core.Mat grayMat = new org.opencv.core.Mat();
            org.opencv.imgproc.Imgproc.cvtColor(convertedMat, grayMat, org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY);

            log.info("Image loaded and converted successfully: " + file.getAbsolutePath());
            log.info("Image size: " + grayMat.size());

            images.add(grayMat);
            labels.add(labelCounter);
            picNames.add(file.getAbsolutePath());
            labelCounter++;
        }

        if (images.isEmpty()) {
            throw new IllegalStateException("No valid images were loaded for training.");
        }

        MatOfInt matLabels = new MatOfInt();
        matLabels.fromList(labels);
        log.info("Training face recognizer with " + images.size() + " images and " + labels.size() + " labels.");
        faceRecognizer.train(images, matLabels);

        return new Face(cascade, faceRecognizer);
    }

    public Mat loadImage(File path) {
        log.info("Loading image from path: " + path.getAbsolutePath());

        if (!path.exists() || !path.isFile()) {
            throw new IllegalArgumentException("File does not exist or is not a valid file: " + path.getAbsolutePath());
        }

        Mat image = opencv_imgcodecs.imread(path.getAbsolutePath());

        if (image.empty()) {
            log.warn("Failed to load image: " + path.getAbsolutePath());
        } else {
            log.info("Loaded image size: " + image.size().toString());
        }
        return image;
    }

    public org.opencv.core.Mat convertMat(org.bytedeco.opencv.opencv_core.Mat bytedecoMat) {
        log.info("Converting Bytedeco Mat to OpenCV Mat");
        log.info("Bytedeco Mat size: " + bytedecoMat.size().toString());

        int size = bytedecoMat.arrayWidth() * bytedecoMat.arrayHeight() * bytedecoMat.channels();
        byte[] byteArray = new byte[size];
        bytedecoMat.data().get(byteArray);

        org.opencv.core.Mat opencvMat = new org.opencv.core.Mat(bytedecoMat.arrayHeight(), bytedecoMat.arrayWidth(), org.opencv.core.CvType.CV_8UC(bytedecoMat.channels()));
        opencvMat.put(0, 0, byteArray);

        if (opencvMat.empty()) {
            log.warn("Converted OpenCV Mat is empty");
        } else {
            log.info("Converted OpenCV Mat size: " + opencvMat.size());
        }

        return opencvMat;
    }
}
