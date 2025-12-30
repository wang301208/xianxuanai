import 'message_type.dart';

/// General-purpose conversation turn used for history timelines.
class ConversationTurn {
  ConversationTurn({
    required this.id,
    required this.role,
    required this.message,
    required this.timestamp,
    this.channel,
    this.attachments,
  });

  final String id;
  final MessageType role;
  final String message;
  final DateTime timestamp;
  final String? channel;
  final Map<String, dynamic>? attachments;

  factory ConversationTurn.fromMap(Map<String, dynamic> map) {
    final roleString = (map['role'] as String? ?? map['author'] as String? ?? 'user').toLowerCase();
    final messageType = roleString.contains('agent') || roleString.contains('assistant')
        ? MessageType.agent
        : MessageType.user;
    return ConversationTurn(
      id: map['id']?.toString() ?? '${map.hashCode}',
      role: messageType,
      message: map['message'] as String? ?? map['content'] as String? ?? '',
      timestamp: _timestampToDate(map['timestamp'] ?? map['createdAt'] ?? map['created_at']) ?? DateTime.now(),
      channel: map['channel'] as String?,
      attachments: map['attachments'] is Map ? Map<String, dynamic>.from(map['attachments'] as Map) : null,
    );
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
    if (value is DateTime) {
      return value;
    }
    return null;
  }
}
