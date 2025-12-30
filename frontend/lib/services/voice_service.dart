import 'voice_service_stub.dart'
    if (dart.library.html) 'voice_service_web.dart';

abstract class VoiceService {
  Future<String?> listenOnce({String locale});
  void speak(String text, {String locale});
  void cancelSpeech();
}

VoiceService createVoiceService() => createVoiceServiceImpl();

