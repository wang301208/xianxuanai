import 'voice_service.dart';

class _NoopVoiceService implements VoiceService {
  @override
  Future<String?> listenOnce({String locale = 'en-US'}) async => null;

  @override
  void speak(String text, {String locale = 'en-US'}) {}

  @override
  void cancelSpeech() {}
}

VoiceService createVoiceServiceImpl() => _NoopVoiceService();

