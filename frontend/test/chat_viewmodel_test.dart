import 'package:auto_gpt_flutter_client/models/chat.dart';
import 'package:auto_gpt_flutter_client/models/message_type.dart';
import 'package:auto_gpt_flutter_client/viewmodels/chat_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/mock_data.dart';
import 'package:flutter_test/flutter_test.dart';

class FakeChatService implements ChatService {
  bool throwOnFetch = false;
  bool throwOnSend = false;

  @override
  Future<List<Chat>> fetchChatsForTask(int taskId) async {
    if (throwOnFetch) {
      throw Exception('fetch error');
    }
    return mockChats.where((c) => c.taskId == taskId).toList();
  }

  @override
  Future<Chat> sendChatMessage(int taskId, String message) async {
    if (throwOnSend) {
      throw Exception('send error');
    }
    return Chat(
      id: mockChats.length + 1,
      taskId: taskId,
      message: 'reply',
      timestamp: DateTime.now(),
      messageType: MessageType.agent,
    );
  }
}

void main() {
  group('ChatViewModel', () {
    late ChatViewModel viewModel;
    late FakeChatService service;

    setUp(() {
      service = FakeChatService();
      viewModel = ChatViewModel(chatService: service);
    });

    test('fetch chats for a specific task', () async {
      await viewModel.fetchChatsForTask(1);
      expect(viewModel.chats.isNotEmpty, true);
      expect(viewModel.chats.every((chat) => chat.taskId == 1), true);
    });

    test('send chat message for a specific task', () async {
      final initialChatsLength = viewModel.chats.length;
      await viewModel.sendChatMessage(1, 'Test message');
      expect(viewModel.chats.length, initialChatsLength + 2);
      expect(viewModel.chats.last.messageType, MessageType.agent);
    });

    test('handles error when fetching chats', () async {
      service.throwOnFetch = true;
      await viewModel.fetchChatsForTask(1);
      expect(viewModel.errorMessage, isNotNull);
      expect(viewModel.chats, isEmpty);
    });

    test('handles error when sending chat message', () async {
      service.throwOnSend = true;
      final initialChatsLength = viewModel.chats.length;
      await viewModel.sendChatMessage(1, 'Test message');
      expect(viewModel.errorMessage, isNotNull);
      expect(viewModel.chats.length, initialChatsLength);
    });
  });
}
