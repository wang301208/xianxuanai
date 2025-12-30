import 'image_picker_service.dart';

class _NoopImagePickerService implements ImagePickerService {
  @override
  Future<PickedImage?> pickImage() async => null;
}

ImagePickerService createImagePickerServiceImpl() => _NoopImagePickerService();

