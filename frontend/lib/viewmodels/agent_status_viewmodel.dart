import 'dart:async';

import 'package:flutter/foundation.dart';

import '../models/agent_status.dart';
import '../services/agent_status_service.dart';

class AgentStatusViewModel extends ChangeNotifier {
  AgentStatusViewModel({required AgentStatusService service}) : _service = service;

  final AgentStatusService _service;

  final List<AgentStatus> agents = [];
  bool isLoading = false;
  String? errorMessage;
  StreamSubscription<List<AgentStatus>>? _subscription;

  Future<void> loadStatuses() async {
    isLoading = true;
    errorMessage = null;
    notifyListeners();
    try {
      final fetched = await _service.fetchStatuses();
      agents
        ..clear()
        ..addAll(fetched);
    } catch (e) {
      errorMessage = e.toString();
    } finally {
      isLoading = false;
      notifyListeners();
    }
  }

  void startListening() {
    _subscription?.cancel();
    _subscription = _service.watchStatuses().listen(
      (event) {
        if (event.isEmpty) {
          return;
        }
        agents
          ..clear()
          ..addAll(event);
        notifyListeners();
      },
      onError: (Object error) {
        errorMessage = error.toString();
        notifyListeners();
      },
    );
  }

  AgentStatus? byId(String id) {
    try {
      return agents.firstWhere((agent) => agent.id == id);
    } catch (_) {
      return null;
    }
  }

  @override
  void dispose() {
    _subscription?.cancel();
    super.dispose();
  }
}
