package mitpc.medsoft.component.service;

import jakarta.annotation.Resource;
import lombok.extern.log4j.Log4j;
import mitpc.medsoft.minio.Minio;
import mitpc.medsoft.model.Face;
import mitpc.medsoft.model.ImageRequest;
import mitpc.medsoft.utils.Response;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.opencv.core.CvType;
import org.opencv.core.Rect;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.File;
import java.util.*;

@Log4j
@Service
public class FaceRecognitionServiceImpl implements FaceRecognitionService {

    @Autowired
    FaceCascadeConfig faceCascadeConfig;
    @Resource
    private Minio minio;
    @Autowired
    private Face faceCascade;

    @Override
    public Response<String> detectFaces(ImageRequest request) {
        String base64Image = request.getImage().split(",")[1];
        byte[] imageBytes = Base64.getDecoder().decode(base64Image);

        Mat image = opencv_imgcodecs.imdecode(new Mat(new BytePointer(imageBytes)), opencv_imgcodecs.IMREAD_COLOR);
        int cropWidth = 160;
        int cropHeight = 160;
        int startX = (image.cols() - cropWidth) / 2;  // Center the crop horizontally
        int startY = (image.rows() - cropHeight) / 2; // Center the crop vertically

        if (startX < 0 || startY < 0 || startX + cropWidth > image.cols() || startY + cropHeight > image.rows()) {
            return Response.toError("Image too small to crop to 160x160 pixels");
        }

        Mat grayscaleImage = new Mat();
        opencv_imgproc.cvtColor(image, grayscaleImage, opencv_imgproc.COLOR_BGR2GRAY);

        RectVector faces = new RectVector();
        faceCascade.getCascade().detectMultiScale(grayscaleImage, faces);
        log.info(faces.size());

        List<String> recognizedNames = new ArrayList<>();
        List<Double> confidenceLevels = new ArrayList<>();
        for (int i = 0; i < faces.size(); i++) {
            Mat faceImage = new Mat(grayscaleImage, faces.get(i));

            // Convert the Mat to the required format for recognition
            org.opencv.core.Mat opencvFaceImage = faceCascadeConfig.convertMat(faceImage);

            // Recognize the face
            int[] label = new int[1];
            double[] confidence = new double[1];
            faceCascade.getFaceRecognizer().predict(opencvFaceImage, label, confidence);

            if (label[0] > 0 && label[0] <= faceCascadeConfig.getPicNames().size()) {
                String fullFilePath = faceCascadeConfig.getPicNames().get(label[0] - 1); // label is 1-based
                String recognizedName = extractFileName(fullFilePath);

                // Ensure confidence level is within 0% to 100%
                double confidencePercentage = Math.min(confidence[0], 100.0); // Cap at 100% if necessary

                recognizedNames.add(recognizedName);
                confidenceLevels.add(confidencePercentage);
            } else {
                recognizedNames.add("Unknown");
                confidenceLevels.add(0.0); // Confidence for unknown faces can be 0
            }
        }

        StringBuilder detectedFacesBuilder = new StringBuilder();
        for (int i = 0; i < recognizedNames.size(); i++) {
            String detectedFace = recognizedNames.get(i) + " (" + String.format("%.2f", confidenceLevels.get(i)) + "%)";
            detectedFacesBuilder.append(detectedFace);
            if (i < recognizedNames.size() - 1) {
                detectedFacesBuilder.append(", ");
            }
        }

        String detectedFaces = detectedFacesBuilder.toString();

        return Response.toSuccess(detectedFaces);
    }


    public String extractFileName(String filePath) {
        // Find the last occurrence of the path separator
        int lastSeparatorIndex = filePath.lastIndexOf(File.separator);

        // Find the last occurrence of the dot for the file extension
        int lastDotIndex = filePath.lastIndexOf('.');

        // Extract the file name without extension
        String fileName = filePath.substring(lastSeparatorIndex + 1, lastDotIndex);

        return fileName;
    }

}
