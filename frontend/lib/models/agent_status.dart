enum AgentLifecycleState {
  idle,
  running,
  waiting,
  error,
  paused,
}

AgentLifecycleState parseAgentState(String? state) {
  final normalized = state?.toLowerCase().trim() ?? '';
  switch (normalized) {
    case 'running':
    case 'active':
    case 'busy':
      return AgentLifecycleState.running;
    case 'waiting':
    case 'blocked':
      return AgentLifecycleState.waiting;
    case 'error':
    case 'failed':
      return AgentLifecycleState.error;
    case 'paused':
      return AgentLifecycleState.paused;
    default:
      return AgentLifecycleState.idle;
  }
}

String agentStateToText(AgentLifecycleState state) {
  switch (state) {
    case AgentLifecycleState.running:
      return 'Active';
    case AgentLifecycleState.waiting:
      return 'Waiting';
    case AgentLifecycleState.error:
      return 'Error';
    case AgentLifecycleState.paused:
      return 'Paused';
    case AgentLifecycleState.idle:
      return 'Idle';
  }
}

/// Snapshot of a single agent/model instance.
class AgentStatus {
  AgentStatus({
    required this.id,
    required this.name,
    required this.state,
    this.currentTask,
    this.cpuUsage,
    this.memoryUsage,
    this.lastUpdated,
    this.statusMessage,
  });

  final String id;
  final String name;
  final AgentLifecycleState state;
  final String? currentTask;
  final double? cpuUsage;
  final double? memoryUsage;
  final DateTime? lastUpdated;
  final String? statusMessage;

  factory AgentStatus.fromMap(Map<String, dynamic> map) {
    return AgentStatus(
      id: map['id']?.toString() ?? '',
      name: map['name'] as String? ?? 'Agent',
      state: parseAgentState(map['state']?.toString()),
      currentTask: map['currentTask'] as String? ?? map['task'] as String?,
      cpuUsage: _parseDouble(map['cpu'] ?? map['cpuUsage'] ?? map['cpu_usage']),
      memoryUsage:
          _parseDouble(map['memory'] ?? map['memoryUsage'] ?? map['memory_usage']),
      lastUpdated: _timestampToDate(map['updatedAt'] ?? map['lastUpdated'] ?? map['last_updated']),
      statusMessage: map['message'] as String? ?? map['statusMessage'] as String?,
    );
  }

  static double? _parseDouble(dynamic value) {
    if (value == null) {
      return null;
    }
    if (value is num) {
      return value.toDouble();
    }
    return double.tryParse(value.toString());
  }

  static DateTime? _timestampToDate(dynamic value) {
    if (value == null) {
      return null;
    }
    if (value is String) {
      return DateTime.tryParse(value);
    }
    if (value is int) {
      return DateTime.fromMillisecondsSinceEpoch(value, isUtc: true).toLocal();
    }
    if (value is double) {
      return DateTime.fromMillisecondsSinceEpoch(value.toInt(), isUtc: true).toLocal();
    }
    return null;
  }

  AgentStatus copyWith({
    AgentLifecycleState? state,
    String? currentTask,
    double? cpuUsage,
    double? memoryUsage,
    DateTime? lastUpdated,
    String? statusMessage,
  }) {
    return AgentStatus(
      id: id,
      name: name,
      state: state ?? this.state,
      currentTask: currentTask ?? this.currentTask,
      cpuUsage: cpuUsage ?? this.cpuUsage,
      memoryUsage: memoryUsage ?? this.memoryUsage,
      lastUpdated: lastUpdated ?? this.lastUpdated,
      statusMessage: statusMessage ?? this.statusMessage,
    );
  }
}
