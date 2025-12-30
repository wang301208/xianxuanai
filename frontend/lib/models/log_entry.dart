/// Represents a single structured log entry emitted by the backend.
class LogEntry {
  LogEntry({
    required this.id,
    required this.timestamp,
    required this.level,
    required this.message,
    this.source,
    this.context = const {},
  });

  final String id;
  final DateTime timestamp;
  final String level;
  final String message;
  final String? source;
  final Map<String, dynamic> context;

  factory LogEntry.fromMap(Map<String, dynamic> map) {
    return LogEntry(
      id: map['id']?.toString() ?? '${map['timestamp'] ?? DateTime.now().millisecondsSinceEpoch}',
      timestamp: _timestampToDate(map['timestamp']) ?? DateTime.now(),
      level: (map['level'] as String? ?? 'INFO').toUpperCase(),
      message: map['message'] as String? ?? '',
      source: map['source'] as String?,
      context: Map<String, dynamic>.from(map['context'] as Map? ?? const {}),
    );
  }

  static DateTime? _timestampToDate(dynamic value) {
    if (value == null) {
      return null;
    }
    if (value is DateTime) {
      return value;
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
}
