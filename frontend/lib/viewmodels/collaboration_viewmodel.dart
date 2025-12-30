import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

class CollaborationViewModel extends ChangeNotifier {
  final WebSocketChannel _channel;

  CollaborationViewModel(WebSocketChannel channel) : _channel = channel {
    _channel.stream.listen(_handleMessage);
  }

  String plan = '';
  String worldModel = '';
  String metrics = '';

  void _handleMessage(dynamic message) {
    try {
      final Map<String, dynamic> data = jsonDecode(message as String);
      if (data.containsKey('plan')) {
        plan = data['plan'] as String? ?? '';
      }
      if (data.containsKey('worldModel')) {
        worldModel = data['worldModel'] as String? ?? '';
      }
      if (data.containsKey('metrics')) {
        metrics = data['metrics'] as String? ?? '';
      }
      notifyListeners();
    } catch (_) {
      // ignore malformed messages
    }
  }

  void sendKnowledge(String text) {
    _channel.sink.add(jsonEncode({'type': 'knowledge', 'content': text}));
  }

  void sendCorrection(String text) {
    _channel.sink.add(jsonEncode({'type': 'correction', 'content': text}));
  }
}
