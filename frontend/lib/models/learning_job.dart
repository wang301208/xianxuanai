/// Represents a self-learning or self-improvement job running on the backend.
class LearningJob {
  LearningJob({
    required this.id,
    required this.name,
    required this.status,
    required this.progress,
    required this.iteration,
    required this.totalIterations,
    this.metrics = const {},
    this.startedAt,
    this.updatedAt,
    this.description,
  });

  final String id;
  final String name;
  final String status;
  final double progress;
  final int iteration;
  final int totalIterations;
  final Map<String, double> metrics;
  final DateTime? startedAt;
  final DateTime? updatedAt;
  final String? description;

  factory LearningJob.fromMap(Map<String, dynamic> map) {
    return LearningJob(
      id: map['id']?.toString() ?? '',
      name: map['name'] as String? ?? 'Unnamed job',
      status: map['status'] as String? ?? 'unknown',
      progress: _parseDouble(map['progress']),
      iteration: map['iteration'] is int ? map['iteration'] as int : int.tryParse('${map['iteration']}') ?? 0,
      totalIterations: map['totalIterations'] is int
          ? map['totalIterations'] as int
          : int.tryParse('${map['totalIterations'] ?? map['total_iterations']}') ?? 0,
      metrics: Map<String, double>.from(
        (map['metrics'] as Map? ?? const {}).map(
          (key, value) => MapEntry(
            key.toString(),
            _parseDouble(value),
          ),
        ),
      ),
      startedAt: _timestampToDate(map['startedAt'] ?? map['started_at']),
      updatedAt: _timestampToDate(map['updatedAt'] ?? map['updated_at']),
      description: map['description'] as String?,
    );
  }

  static double _parseDouble(dynamic value) {
    if (value == null) {
      return 0.0;
    }
    if (value is num) {
      return value.toDouble();
    }
    return double.tryParse(value.toString()) ?? 0.0;
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

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'name': name,
      'status': status,
      'progress': progress,
      'iteration': iteration,
      'totalIterations': totalIterations,
      'metrics': metrics,
      'startedAt': startedAt?.toIso8601String(),
      'updatedAt': updatedAt?.toIso8601String(),
      if (description != null) 'description': description,
    };
  }
}
