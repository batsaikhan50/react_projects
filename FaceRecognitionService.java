package mitpc.medsoft.component.service;

import mitpc.medsoft.model.ImageRequest;
import mitpc.medsoft.utils.Response;

public interface FaceRecognitionService {
    Response<String> detectFaces(ImageRequest request);
}
