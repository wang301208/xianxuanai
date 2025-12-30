import 'package:flutter/foundation.dart';

import '../models/chat.dart';
import '../models/message_type.dart';

abstract class ChatService {
  Future<List<Chat>> fetchChatsForTask(int taskId);
  Future<Chat> sendChatMessage(int taskId, String message);
}

class ChatViewModel extends ChangeNotifier {
  final ChatService _chatService;

  ChatViewModel({required ChatService chatService}) : _chatService = chatService;

  final List<Chat> chats = [];
  String? errorMessage;

  Future<void> fetchChatsForTask(int taskId) async {
    try {
      errorMessage = null;
      final fetched = await _chatService.fetchChatsForTask(taskId);
      chats
        ..clear()
        ..addAll(fetched);
      notifyListeners();
    } catch (e) {
      errorMessage = e.toString();
      notifyListeners();
    }
  }

  Future<void> sendChatMessage(int taskId, String message) async {
    try {
      errorMessage = null;
      final agentChat = await _chatService.sendChatMessage(taskId, message);
      final userChat = Chat(
        id: chats.isEmpty ? 1 : chats.last.id + 1,
        taskId: taskId,
        message: message,
        timestamp: DateTime.now(),
        messageType: MessageType.user,
      );
      chats.addAll([userChat, agentChat]);
      notifyListeners();
    } catch (e) {
      errorMessage = e.toString();
      notifyListeners();
    }
  }
}
