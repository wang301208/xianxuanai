import 'dart:async';
import 'dart:html' as html;
import 'dart:js' as js;

import 'voice_service.dart';

class _WebVoiceService implements VoiceService {
  @override
  Future<String?> listenOnce({String locale = 'en-US'}) {
    final completer = Completer<String?>();
    try {
      final dynamic speechRecognition = js.context['SpeechRecognition'] ?? js.context['webkitSpeechRecognition'];
      if (speechRecognition == null) {
        completer.complete(null);
        return completer.future;
      }

      final dynamic recognizer = js.JsObject(speechRecognition);
      recognizer['lang'] = locale;
      recognizer['interimResults'] = false;
      recognizer['maxAlternatives'] = 1;

      recognizer['onresult'] = js.allowInterop((dynamic event) {
        try {
          final dynamic results = event['results'];
          final dynamic first = results[0];
          final dynamic alt = first[0];
          final String transcript = (alt['transcript'] as String?)?.trim() ?? '';
          if (!completer.isCompleted) {
            completer.complete(transcript.isEmpty ? null : transcript);
          }
        } catch (_) {
          if (!completer.isCompleted) {
            completer.complete(null);
          }
        }
      });
      recognizer['onerror'] = js.allowInterop((dynamic _) {
        if (!completer.isCompleted) {
          completer.complete(null);
        }
      });
      recognizer['onend'] = js.allowInterop(() {
        if (!completer.isCompleted) {
          completer.complete(null);
        }
      });

      recognizer.callMethod('start');
    } catch (_) {
      if (!completer.isCompleted) {
        completer.complete(null);
      }
    }
    return completer.future;
  }

  @override
  void speak(String text, {String locale = 'en-US'}) {
    final trimmed = text.trim();
    if (trimmed.isEmpty) {
      return;
    }
    try {
      final synth = html.window.speechSynthesis;
      if (synth == null) {
        return;
      }
      synth.cancel();
      final utterance = html.SpeechSynthesisUtterance(trimmed);
      utterance.lang = locale;
      synth.speak(utterance);
    } catch (_) {}
  }

  @override
  void cancelSpeech() {
    try {
      html.window.speechSynthesis?.cancel();
    } catch (_) {}
  }
}

VoiceService createVoiceServiceImpl() => _WebVoiceService();

