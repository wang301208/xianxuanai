import 'dart:async';
import 'dart:convert';

import 'package:web_socket_channel/web_socket_channel.dart';

import '../models/conversation_turn.dart';
import '../models/log_entry.dart';

import 'api_client.dart';
import 'agent_status_service.dart';

abstract class LogService {
  Future<List<LogEntry>> fetchLogs({String? level, int? limit});
  Future<List<ConversationTurn>> fetchConversation({int? limit});
  Future<ConversationTurn> sendChat({
    required String sessionId,
    required List<Map<String, String>> messages,
    String? imageBase64,
    String? imageMime,
  });
  Stream<LogEntry> watchLogs();
  Stream<ConversationTurn> watchConversation();
}

class HttpLogService implements LogService {
  HttpLogService({
    ApiClient? apiClient,
    this.logsPath = '/api/logs/system',
    this.conversationPath = '/api/conversations/history',
    this.chatPath = '/api/chat',
    String logsSocketPath = '/ws/logs/system',
    String conversationSocketPath = '/ws/logs/conversation',
    WebSocketChannelFactory? logsChannelFactory,
    WebSocketChannelFactory? conversationChannelFactory,
  })  : _client = apiClient ?? ApiClient(),
        _logsSocketPath = logsSocketPath,
        _conversationSocketPath = conversationSocketPath,
        _logsChannelFactory = logsChannelFactory,
        _conversationChannelFactory = conversationChannelFactory;

  final ApiClient _client;
  final String logsPath;
  final String conversationPath;
  final String chatPath;
  final String _logsSocketPath;
  final String _conversationSocketPath;
  final WebSocketChannelFactory? _logsChannelFactory;
  final WebSocketChannelFactory? _conversationChannelFactory;

  @override
  Future<List<LogEntry>> fetchLogs({String? level, int? limit}) async {
    final data = await _client.getJsonList(
      logsPath,
      query: {
        if (level != null && level.isNotEmpty) 'level': level,
        if (limit != null) 'limit': limit,
      },
    );
    return data
        .whereType<Map<String, dynamic>>()
        .map(LogEntry.fromMap)
        .toList(growable: false);
  }

  @override
  Future<List<ConversationTurn>> fetchConversation({int? limit}) async {
    final data = await _client.getJsonList(
      conversationPath,
      query: {
        if (limit != null) 'limit': limit,
      },
    );
    return data
        .whereType<Map<String, dynamic>>()
        .map(ConversationTurn.fromMap)
        .toList(growable: false);
  }

  @override
  Future<ConversationTurn> sendChat({
    required String sessionId,
    required List<Map<String, String>> messages,
    String? imageBase64,
    String? imageMime,
  }) async {
    final data = await _client.postJson(
      chatPath,
      body: {
        'session_id': sessionId,
        'messages': messages,
        if (imageBase64 != null && imageBase64.isNotEmpty) 'image_base64': imageBase64,
        if (imageMime != null && imageMime.isNotEmpty) 'image_mime': imageMime,
      },
    );
    return ConversationTurn.fromMap(
      {
        'id': data['turn_id']?.toString() ?? DateTime.now().millisecondsSinceEpoch.toString(),
        'role': 'agent',
        'message': data['reply']?.toString() ?? '',
        'timestamp': DateTime.now().toIso8601String(),
        'channel': 'chat',
        'session_id': data['session_id']?.toString() ?? sessionId,
      },
    );
  }

  @override
  Stream<LogEntry> watchLogs() {
    final channel = _createChannel(_logsChannelFactory, _logsSocketPath);
    if (channel == null) {
      return const Stream<LogEntry>.empty();
    }
    final controller = StreamController<LogEntry>.broadcast();
    late final StreamSubscription subscription;
    subscription = channel.stream.listen(
      (dynamic event) {
        try {
          final payload = event is String ? jsonDecode(event) : event;
          if (payload is Map<String, dynamic>) {
            controller.add(LogEntry.fromMap(payload));
          }
        } catch (_) {
          // ignore malformed log payloads
        }
      },
      onError: (_) {},
      onDone: () {
        if (!controller.isClosed) {
          controller.close();
        }
      },
    );
    controller.onCancel = () async {
      await subscription.cancel();
      await channel.sink.close();
    };
    return controller.stream;
  }

  @override
  Stream<ConversationTurn> watchConversation() {
    final channel = _createChannel(_conversationChannelFactory, _conversationSocketPath);
    if (channel == null) {
      return const Stream<ConversationTurn>.empty();
    }
    final controller = StreamController<ConversationTurn>.broadcast();
    late final StreamSubscription subscription;
    subscription = channel.stream.listen(
      (dynamic event) {
        try {
          final payload = event is String ? jsonDecode(event) : event;
          if (payload is Map<String, dynamic>) {
            controller.add(ConversationTurn.fromMap(payload));
          }
        } catch (_) {
          // ignore malformed payloads
        }
      },
      onError: (_) {},
      onDone: () {
        if (!controller.isClosed) {
          controller.close();
        }
      },
    );
    controller.onCancel = () async {
      await subscription.cancel();
      await channel.sink.close();
    };
    return controller.stream;
  }

  WebSocketChannel? _createChannel(
    WebSocketChannelFactory? factory,
    String socketPath,
  ) {
    if (factory != null) {
      try {
        return factory();
      } catch (_) {
        return null;
      }
    }
    final base = _client.baseUri;
    final scheme = base.scheme == 'https' ? 'wss' : 'ws';
    final path = socketPath.startsWith('/') ? socketPath : '/$socketPath';
    final uri = Uri(
      scheme: scheme,
      userInfo: base.userInfo,
      host: base.host,
      port: base.hasPort ? base.port : null,
      path: path,
    );
    try {
      return WebSocketChannel.connect(uri);
    } catch (_) {
      return null;
    }
  }
}
