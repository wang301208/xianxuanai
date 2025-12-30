import 'dart:async';
import 'dart:convert';
import 'dart:html' as html;

import 'image_picker_service.dart';

class _WebImagePickerService implements ImagePickerService {
  @override
  Future<PickedImage?> pickImage() async {
    final completer = Completer<PickedImage?>();
    try {
      final input = html.FileUploadInputElement()..accept = 'image/*';
      input.click();
      input.onChange.first.then((_) {
        final files = input.files;
        if (files == null || files.isEmpty) {
          if (!completer.isCompleted) completer.complete(null);
          return;
        }
        final file = files.first;
        final reader = html.FileReader();
        reader.readAsArrayBuffer(file);
        reader.onError.first.then((_) {
          if (!completer.isCompleted) completer.complete(null);
        });
        reader.onLoadEnd.first.then((_) {
          try {
            final data = reader.result;
            if (data is! ByteBuffer) {
              if (!completer.isCompleted) completer.complete(null);
              return;
            }
            final bytes = data.asUint8List();
            final base64 = base64Encode(bytes);
            final mime = file.type.isNotEmpty ? file.type : 'image/png';
            if (!completer.isCompleted) {
              completer.complete(
                PickedImage(
                  base64: base64,
                  mimeType: mime,
                  name: file.name,
                ),
              );
            }
          } catch (_) {
            if (!completer.isCompleted) completer.complete(null);
          }
        });
      });
    } catch (_) {
      if (!completer.isCompleted) completer.complete(null);
    }
    return completer.future;
  }
}

ImagePickerService createImagePickerServiceImpl() => _WebImagePickerService();

