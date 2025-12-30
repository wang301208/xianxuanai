import 'message_type.dart';

class Chat {
  final int id;
  final int taskId;
  final String message;
  final DateTime timestamp;
  final MessageType messageType;

  Chat({
    required this.id,
    required this.taskId,
    required this.message,
    required this.timestamp,
    required this.messageType,
  });

  factory Chat.fromMap(Map<String, dynamic> map) {
    return Chat(
      id: map['id'] as int,
      taskId: map['taskId'] as int,
      message: map['message'] as String,
      timestamp: DateTime.parse(map['timestamp'] as String),
      messageType: MessageType.values.firstWhere(
        (e) => e.name == map['messageType'],
      ),
    );
  }

  Map<String, dynamic> toMap() => {
        'id': id,
        'taskId': taskId,
        'message': message,
        'timestamp': timestamp.toIso8601String(),
        'messageType': messageType.name,
      };

  @override
  bool operator ==(Object other) {
    return other is Chat &&
        other.id == id &&
        other.taskId == taskId &&
        other.message == message &&
        other.timestamp == timestamp &&
        other.messageType == messageType;
  }

  @override
  int get hashCode => Object.hash(id, taskId, message, timestamp, messageType);

  @override
  String toString() =>
      'Chat(id: $id, taskId: $taskId, message: $message, timestamp: $timestamp, messageType: $messageType)';
}
