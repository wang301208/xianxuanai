import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:uuid/uuid.dart';

import '../models/conversation_turn.dart';
import '../models/log_entry.dart';
import '../services/log_service.dart';
import '../services/voice_service.dart';
import '../services/image_picker_service.dart';

class LogViewModel extends ChangeNotifier {
  LogViewModel({required LogService service}) : _service = service;

  final LogService _service;
  final VoiceService _voice = createVoiceService();
  final ImagePickerService _images = createImagePickerService();

  final List<LogEntry> logs = [];
  final List<ConversationTurn> conversation = [];
  final String sessionId = const Uuid().v4();
  final List<Map<String, String>> _sessionMessages = [];
  bool ttsEnabled = false;
  bool isLoadingLogs = false;
  bool isLoadingConversation = false;
  String? errorMessage;
  String? _levelFilter;

  StreamSubscription<LogEntry>? _logSubscription;
  StreamSubscription<ConversationTurn>? _conversationSubscription;

  Future<void> loadLogs({String? level}) async {
    isLoadingLogs = true;
    errorMessage = null;
    notifyListeners();
    try {
      final fetched = await _service.fetchLogs(level: level);
      logs
        ..clear()
        ..addAll(fetched);
      _levelFilter = level;
    } catch (e) {
      errorMessage = e.toString();
    } finally {
      isLoadingLogs = false;
      notifyListeners();
    }
  }

  Future<void> loadConversation() async {
    isLoadingConversation = true;
    errorMessage = null;
    notifyListeners();
    try {
      final fetched = await _service.fetchConversation();
      conversation
        ..clear()
        ..addAll(fetched);
    } catch (e) {
      errorMessage = e.toString();
    } finally {
      isLoadingConversation = false;
      notifyListeners();
    }
  }

  Future<void> sendChatMessage(String message) async {
    final trimmed = message.trim();
    if (trimmed.isEmpty) {
      return;
    }

    errorMessage = null;
    final now = DateTime.now();
    final userTurn = ConversationTurn.fromMap(
      {
        'id': now.microsecondsSinceEpoch.toString(),
        'role': 'user',
        'message': trimmed,
        'timestamp': now.toIso8601String(),
        'channel': 'chat',
        'session_id': sessionId,
      },
    );
    conversation.insert(0, userTurn);
    _sessionMessages.add({'role': 'user', 'content': trimmed});
    if (_sessionMessages.length > 60) {
      _sessionMessages.removeRange(0, _sessionMessages.length - 60);
    }
    notifyListeners();

    try {
      final reply = await _service.sendChat(
        sessionId: sessionId,
        messages: List<Map<String, String>>.from(_sessionMessages),
      );
      conversation.insert(0, reply);
      _sessionMessages.add({'role': 'assistant', 'content': reply.message});
      if (_sessionMessages.length > 60) {
        _sessionMessages.removeRange(0, _sessionMessages.length - 60);
      }
      if (ttsEnabled) {
        _voice.speak(reply.message, locale: 'zh-CN');
      }
    } catch (e) {
      errorMessage = e.toString();
    } finally {
      notifyListeners();
    }
  }

  Future<void> sendChatImage() async {
    errorMessage = null;
    notifyListeners();

    final picked = await _images.pickImage();
    if (picked == null) {
      errorMessage = 'No image selected or image picking not supported.';
      notifyListeners();
      return;
    }

    final now = DateTime.now();
    final userTurn = ConversationTurn.fromMap(
      {
        'id': now.microsecondsSinceEpoch.toString(),
        'role': 'user',
        'message': '(image) ${picked.name}',
        'timestamp': now.toIso8601String(),
        'channel': 'chat',
        'session_id': sessionId,
        'attachments': {
          'image': {'base64': picked.base64, 'mime': picked.mimeType},
        },
      },
    );
    conversation.insert(0, userTurn);
    _sessionMessages.add({'role': 'user', 'content': userTurn.message});
    if (_sessionMessages.length > 60) {
      _sessionMessages.removeRange(0, _sessionMessages.length - 60);
    }
    notifyListeners();

    try {
      final reply = await _service.sendChat(
        sessionId: sessionId,
        messages: List<Map<String, String>>.from(_sessionMessages),
        imageBase64: picked.base64,
        imageMime: picked.mimeType,
      );
      conversation.insert(0, reply);
      _sessionMessages.add({'role': 'assistant', 'content': reply.message});
      if (_sessionMessages.length > 60) {
        _sessionMessages.removeRange(0, _sessionMessages.length - 60);
      }
      if (ttsEnabled) {
        _voice.speak(reply.message, locale: 'zh-CN');
      }
    } catch (e) {
      errorMessage = e.toString();
      notifyListeners();
    }
  }

  Future<void> startVoiceInput() async {
    errorMessage = null;
    notifyListeners();

    final transcript = await _voice.listenOnce(locale: 'zh-CN');
    if (transcript == null || transcript.trim().isEmpty) {
      errorMessage = 'Voice recognition not available or no speech captured.';
      notifyListeners();
      return;
    }
    await sendChatMessage(transcript);
  }

  void toggleTts() {
    ttsEnabled = !ttsEnabled;
    if (!ttsEnabled) {
      _voice.cancelSpeech();
    }
    notifyListeners();
  }

  void startStreaming() {
    _logSubscription?.cancel();
    _logSubscription = _service.watchLogs().listen(
      (event) {
        if (_levelFilter != null && event.level != _levelFilter) {
          return;
        }
        logs.insert(0, event);
        if (logs.length > 200) {
          logs.removeRange(200, logs.length);
        }
        notifyListeners();
      },
      onError: (Object error) {
        errorMessage = error.toString();
        notifyListeners();
      },
    );

    _conversationSubscription?.cancel();
    _conversationSubscription = _service.watchConversation().listen(
      (event) {
        conversation.insert(0, event);
        if (conversation.length > 200) {
          conversation.removeRange(200, conversation.length);
        }
        notifyListeners();
      },
      onError: (Object error) {
        errorMessage = error.toString();
        notifyListeners();
      },
    );
  }

  String? get levelFilter => _levelFilter;

  @override
  void dispose() {
    _logSubscription?.cancel();
    _conversationSubscription?.cancel();
    _voice.cancelSpeech();
    super.dispose();
  }
}
