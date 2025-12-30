import '../models/task.dart';
import '../models/chat.dart';
import '../models/message_type.dart';

List<Task> mockTasks = [
  Task(id: 1, title: 'Sample Task 1'),
  Task(id: 2, title: 'Sample Task 2'),
];

List<Chat> mockChats = [
  Chat(
      id: 1,
      taskId: 1,
      message: 'Hello',
      timestamp: DateTime(2023, 1, 1),
      messageType: MessageType.user),
  Chat(
      id: 2,
      taskId: 1,
      message: 'Hi there',
      timestamp: DateTime(2023, 1, 1),
      messageType: MessageType.agent),
];
