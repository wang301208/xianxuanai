import 'dart:async';
import 'dart:convert';

import 'package:web_socket_channel/web_socket_channel.dart';

import '../models/agent_status.dart';

import 'api_client.dart';

abstract class AgentStatusService {
  Future<List<AgentStatus>> fetchStatuses();
  Stream<List<AgentStatus>> watchStatuses();
}

typedef WebSocketChannelFactory = WebSocketChannel Function();

class HttpAgentStatusService implements AgentStatusService {
  HttpAgentStatusService({
    ApiClient? apiClient,
    this.basePath = '/api/agents/status',
    String websocketPath = '/ws/agents/status',
    WebSocketChannelFactory? channelFactory,
  })  : _client = apiClient ?? ApiClient(),
        _websocketPath = websocketPath,
        _channelFactory = channelFactory;

  final ApiClient _client;
  final String basePath;
  final String _websocketPath;
  final WebSocketChannelFactory? _channelFactory;

  @override
  Future<List<AgentStatus>> fetchStatuses() async {
    final data = await _client.getJsonList(basePath);
    return data
        .whereType<Map<String, dynamic>>()
        .map(AgentStatus.fromMap)
        .toList(growable: false);
  }

  @override
  Stream<List<AgentStatus>> watchStatuses() {
    final channel = _createChannel();
    if (channel == null) {
      return const Stream<List<AgentStatus>>.empty();
    }
    final controller = StreamController<List<AgentStatus>>.broadcast();
    late final StreamSubscription subscription;
    subscription = channel.stream.listen(
      (dynamic event) {
        try {
          final payload = event is String ? jsonDecode(event) : event;
          if (payload is Map<String, dynamic> && payload.containsKey('agents')) {
            final agents = (payload['agents'] as List<dynamic>)
                .whereType<Map<String, dynamic>>()
                .map(AgentStatus.fromMap)
                .toList(growable: false);
            controller.add(agents);
          } else if (payload is List<dynamic>) {
            final agents = payload
                .whereType<Map<String, dynamic>>()
                .map(AgentStatus.fromMap)
                .toList(growable: false);
            controller.add(agents);
          }
        } catch (_) {
          // Ignore malformed payloads; logging handled on the server.
        }
      },
      onError: (_) {},
      onDone: () {
        if (!controller.isClosed) {
          controller.close();
        }
      },
      cancelOnError: false,
    );

    controller.onCancel = () async {
      await subscription.cancel();
      await channel.sink.close();
    };

    return controller.stream;
  }

  WebSocketChannel? _createChannel() {
    if (_channelFactory != null) {
      try {
        return _channelFactory!.call();
      } catch (_) {
        return null;
      }
    }

    final base = _client.baseUri;
    final scheme = base.scheme == 'https' ? 'wss' : 'ws';
    final path = _websocketPath.startsWith('/')
        ? _websocketPath
        : '/$_websocketPath';
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
