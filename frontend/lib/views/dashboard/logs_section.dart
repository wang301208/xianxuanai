import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'dart:convert';

import '../../models/conversation_turn.dart';
import '../../models/log_entry.dart';
import '../../models/message_type.dart';
import '../../viewmodels/log_viewmodel.dart';
import '../chat/chat_input_field.dart';
import '../shared/empty_state.dart';
import '../shared/panel_container.dart';
import '../shared/section_header.dart';

class LogsSection extends StatelessWidget {
  const LogsSection({super.key});

  static const _levels = ['ALL', 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'];

  @override
  Widget build(BuildContext context) {
    return Consumer<LogViewModel>(
      builder: (context, viewModel, _) {
        final activeLevel = viewModel.levelFilter?.toUpperCase() ?? 'ALL';
        return DefaultTabController(
          length: 3,
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                SectionHeader(
                  icon: Icons.history_edu,
                  title: 'Logs & Conversation',
                  subtitle: 'Audit what the agent said and why, with contextual system logs.',
                  actions: [
                    DropdownButton<String>(
                      value: activeLevel,
                      underline: const SizedBox.shrink(),
                      items: _levels
                          .map(
                            (level) => DropdownMenuItem<String>(
                              value: level,
                              child: Text('Level: $level'),
                            ),
                          )
                          .toList(growable: false),
                      onChanged: (value) {
                        final target = value == null || value == 'ALL' ? null : value;
                        viewModel.loadLogs(level: target);
                      },
                    ),
                    IconButton(
                      tooltip: 'Refresh logs & conversation',
                      icon: const Icon(Icons.refresh),
                      onPressed: () async {
                        await Future.wait([
                          viewModel.loadLogs(level: viewModel.levelFilter),
                          viewModel.loadConversation(),
                        ]);
                      },
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                PanelContainer(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      TabBar(
                        labelColor: Theme.of(context).colorScheme.primary,
                        tabs: const [
                          Tab(icon: Icon(Icons.chat_bubble_outline), text: 'Live Chat'),
                          Tab(icon: Icon(Icons.chat), text: 'Conversation'),
                          Tab(icon: Icon(Icons.article_outlined), text: 'System Logs'),
                        ],
                      ),
                      const SizedBox(height: 12),
                      if (viewModel.isLoadingLogs || viewModel.isLoadingConversation)
                        const LinearProgressIndicator(),
                      const SizedBox(height: 12),
                      Expanded(
                        child: TabBarView(
                          children: [
                            const _LiveChatPanel(),
                            _ConversationTimeline(turns: viewModel.conversation),
                            _SystemLogsList(entries: viewModel.logs),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

class _LiveChatPanel extends StatelessWidget {
  const _LiveChatPanel();

  @override
  Widget build(BuildContext context) {
    return Consumer<LogViewModel>(
      builder: (context, viewModel, _) {
        final chatTurns = viewModel.conversation
            .where((turn) => (turn.channel ?? '') == 'chat' || turn.channel == null)
            .toList(growable: false);
        final theme = Theme.of(context);

        return Column(
          children: [
            Row(
              children: [
                const Spacer(),
                IconButton(
                  tooltip: viewModel.ttsEnabled ? 'Disable TTS' : 'Enable TTS',
                  icon: Icon(viewModel.ttsEnabled ? Icons.volume_up : Icons.volume_off),
                  onPressed: viewModel.toggleTts,
                ),
              ],
            ),
            if (viewModel.errorMessage != null && viewModel.errorMessage!.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Text(
                  viewModel.errorMessage!,
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.error,
                  ),
                ),
              ),
            Expanded(
              child: chatTurns.isEmpty
                  ? const Center(
                      child: EmptyState(
                        icon: Icons.chat_bubble_outline,
                        title: 'Start a conversation',
                        message: 'Send a message to begin multi-turn chat with the agent.',
                      ),
                    )
                  : ListView.builder(
                      reverse: true,
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                      itemCount: chatTurns.length,
                      itemBuilder: (context, index) {
                        final turn = chatTurns[index];
                        final alignment = turn.role == MessageType.user
                            ? Alignment.centerRight
                            : Alignment.centerLeft;
                        final bubbleColor = turn.role == MessageType.user
                            ? theme.colorScheme.primaryContainer
                            : theme.colorScheme.secondaryContainer;
                        final icon = turn.role == MessageType.user
                            ? Icons.person_outline
                            : Icons.smart_toy;
                        return Align(
                          alignment: alignment,
                          child: ConstrainedBox(
                            constraints: const BoxConstraints(maxWidth: 640),
                            child: Card(
                              color: bubbleColor,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.only(
                                  topLeft: const Radius.circular(16),
                                  topRight: const Radius.circular(16),
                                  bottomLeft: Radius.circular(
                                      turn.role == MessageType.user ? 16 : 4),
                                  bottomRight: Radius.circular(
                                      turn.role == MessageType.user ? 4 : 16),
                                ),
                              ),
                              child: Padding(
                                padding: const EdgeInsets.all(12),
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Row(
                                      children: [
                                        Icon(icon, size: 16),
                                        const SizedBox(width: 6),
                                        Text(turn.role == MessageType.user ? 'You' : 'Agent'),
                                        const Spacer(),
                                        Text(
                                          _formatConversationTimestamp(turn.timestamp),
                                          style: theme.textTheme.bodySmall,
                                        ),
                                      ],
                                    ),
                                    const SizedBox(height: 8),
                                    Text(turn.message),
                                    _AttachmentView(attachments: turn.attachments),
                                  ],
                                ),
                              ),
                            ),
                          ),
                        );
                      },
                    ),
            ),
            const SizedBox(height: 8),
            ChatInputField(
              onSendPressed: (text) => viewModel.sendChatMessage(text),
              onVoicePressed: () => viewModel.startVoiceInput(),
              onImagePressed: () => viewModel.sendChatImage(),
            ),
          ],
        );
      },
    );
  }

  String _formatConversationTimestamp(DateTime timestamp) {
    final local = timestamp.toLocal();
    final hour = local.hour.toString().padLeft(2, '0');
    final minute = local.minute.toString().padLeft(2, '0');
    return '$hour:$minute';
  }
}

class _AttachmentView extends StatelessWidget {
  const _AttachmentView({required this.attachments});

  final Map<String, dynamic>? attachments;

  @override
  Widget build(BuildContext context) {
    if (attachments == null || attachments!.isEmpty) {
      return const SizedBox.shrink();
    }
    final image = attachments!['image'];
    if (image is! Map) {
      return const SizedBox.shrink();
    }
    final base64 = image['base64']?.toString();
    if (base64 == null || base64.isEmpty) {
      return const SizedBox.shrink();
    }
    try {
      final bytes = base64Decode(base64);
      return Padding(
        padding: const EdgeInsets.only(top: 10),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(12),
          child: Image.memory(
            bytes,
            width: 360,
            fit: BoxFit.cover,
          ),
        ),
      );
    } catch (_) {
      return const SizedBox.shrink();
    }
  }
}

class _ConversationTimeline extends StatelessWidget {
  const _ConversationTimeline({required this.turns});

  final List<ConversationTurn> turns;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return RefreshIndicator(
      onRefresh: () {
        final viewModel = context.read<LogViewModel>();
        return viewModel.loadConversation();
      },
      child: turns.isEmpty
          ? const SingleChildScrollView(
              physics: AlwaysScrollableScrollPhysics(),
              child: SizedBox(
                height: 240,
                child: EmptyState(
                  icon: Icons.chat_bubble_outline,
                  title: 'No conversation history',
                  message: 'Interact with the agent to start building a shared timeline.',
                ),
              ),
            )
          : ListView.separated(
              physics: const AlwaysScrollableScrollPhysics(),
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              itemCount: turns.length,
              separatorBuilder: (_, __) => const SizedBox(height: 12),
              itemBuilder: (context, index) {
                final turn = turns[index];
                final alignment =
                    turn.role == MessageType.user ? Alignment.centerRight : Alignment.centerLeft;
                final bubbleColor = turn.role == MessageType.user
                    ? theme.colorScheme.primaryContainer
                    : theme.colorScheme.secondaryContainer;
                final icon = turn.role == MessageType.user ? Icons.person_outline : Icons.smart_toy;
                return Align(
                  alignment: alignment,
                  child: ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: 520),
                    child: Card(
                      color: bubbleColor,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.only(
                          topLeft: const Radius.circular(16),
                          topRight: const Radius.circular(16),
                          bottomLeft: Radius.circular(turn.role == MessageType.user ? 16 : 4),
                          bottomRight: Radius.circular(turn.role == MessageType.user ? 4 : 16),
                        ),
                      ),
                      child: Padding(
                        padding: const EdgeInsets.all(16),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                Icon(icon, size: 16),
                                const SizedBox(width: 6),
                                Text(turn.role == MessageType.user ? 'User' : 'Agent'),
                                const SizedBox(width: 8),
                                Text(
                                  _formatConversationTimestamp(turn.timestamp),
                                  style: theme.textTheme.bodySmall,
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                            Text(turn.message),
                            if (turn.channel != null && turn.channel!.isNotEmpty) ...[
                              const SizedBox(height: 8),
                              Chip(
                                avatar: const Icon(Icons.alternate_email, size: 16),
                                label: Text(turn.channel!),
                              ),
                            ],
                          ],
                        ),
                      ),
                    ),
                  ),
                );
              },
            ),
    );
  }

  String _formatConversationTimestamp(DateTime timestamp) {
    final local = timestamp.toLocal();
    final hour = local.hour.toString().padLeft(2, '0');
    final minute = local.minute.toString().padLeft(2, '0');
    return '$hour:$minute';
  }
}

class _SystemLogsList extends StatelessWidget {
  const _SystemLogsList({required this.entries});

  final List<LogEntry> entries;

  @override
  Widget build(BuildContext context) {
    return RefreshIndicator(
      onRefresh: () {
        final viewModel = context.read<LogViewModel>();
        return viewModel.loadLogs(level: viewModel.levelFilter);
      },
      child: entries.isEmpty
          ? const SingleChildScrollView(
              physics: AlwaysScrollableScrollPhysics(),
              child: SizedBox(
                height: 240,
                child: EmptyState(
                  icon: Icons.article_outlined,
                  title: 'No system logs',
                  message: 'Run agent tasks to collect execution traces and diagnostics.',
                ),
              ),
            )
          : ListView.separated(
              physics: const AlwaysScrollableScrollPhysics(),
              itemCount: entries.length,
              separatorBuilder: (_, __) => const Divider(height: 1),
              itemBuilder: (context, index) {
                final entry = entries[index];
                return ListTile(
                  leading: _LevelBadge(level: entry.level),
                  title: Text(entry.message),
                  subtitle: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Source: ${entry.source ?? 'unknown'}'),
                      if (entry.context.isNotEmpty)
                        Padding(
                          padding: const EdgeInsets.only(top: 4),
                          child: Wrap(
                            spacing: 6,
                            runSpacing: 6,
                            children: entry.context.entries
                                .map(
                                  (e) => Chip(
                                    label: Text('${e.key}: ${e.value}'),
                                    visualDensity: VisualDensity.compact,
                                  ),
                                )
                                .toList(growable: false),
                          ),
                        ),
                    ],
                  ),
                  trailing: Text(_formatLogTimestamp(entry.timestamp)),
                );
              },
            ),
    );
  }

  String _formatLogTimestamp(DateTime timestamp) {
    final local = timestamp.toLocal();
    final hour = local.hour.toString().padLeft(2, '0');
    final minute = local.minute.toString().padLeft(2, '0');
    final second = local.second.toString().padLeft(2, '0');
    return '$hour:$minute:$second';
  }
}

class _LevelBadge extends StatelessWidget {
  const _LevelBadge({required this.level});

  final String level;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final normalized = level.toUpperCase();
    Color palette;
    switch (normalized) {
      case 'ERROR':
      case 'CRITICAL':
        palette = theme.colorScheme.errorContainer;
        break;
      case 'WARN':
      case 'WARNING':
        palette = theme.colorScheme.tertiaryContainer;
        break;
      case 'DEBUG':
        palette = theme.colorScheme.surfaceVariant;
        break;
      default:
        palette = theme.colorScheme.secondaryContainer;
    }
    return Chip(
      label: Text(normalized),
      backgroundColor: palette,
    );
  }
}
