import 'image_picker_service_stub.dart'
    if (dart.library.html) 'image_picker_service_web.dart';

class PickedImage {
  PickedImage({
    required this.base64,
    required this.mimeType,
    required this.name,
  });

  final String base64;
  final String mimeType;
  final String name;
}

abstract class ImagePickerService {
  Future<PickedImage?> pickImage();
}

ImagePickerService createImagePickerService() => createImagePickerServiceImpl();

