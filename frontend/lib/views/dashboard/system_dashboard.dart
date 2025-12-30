import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../viewmodels/agent_status_viewmodel.dart';
import '../../viewmodels/knowledge_base_viewmodel.dart';
import '../../viewmodels/learning_progress_viewmodel.dart';
import '../../viewmodels/log_viewmodel.dart';
import '../../viewmodels/memory_viewmodel.dart';
import 'agent_status_section.dart';
import 'knowledge_base_section.dart';
import 'learning_progress_section.dart';
import 'logs_section.dart';
import 'memory_management_section.dart';

class SystemDashboard extends StatefulWidget {
  const SystemDashboard({
    super.key,
    required this.knowledgeViewModel,
    required this.memoryViewModel,
    required this.learningViewModel,
    required this.agentStatusViewModel,
    required this.logViewModel,
  });

  final KnowledgeBaseViewModel knowledgeViewModel;
  final MemoryViewModel memoryViewModel;
  final LearningProgressViewModel learningViewModel;
  final AgentStatusViewModel agentStatusViewModel;
  final LogViewModel logViewModel;

  @override
  State<SystemDashboard> createState() => _SystemDashboardState();
}

class _SystemDashboardState extends State<SystemDashboard> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      widget.knowledgeViewModel.loadEntries();
      widget.memoryViewModel.loadRecent();
      widget.learningViewModel
        ..loadJobs()
        ..startStreaming();
      widget.agentStatusViewModel
        ..loadStatuses()
        ..startListening();
      widget.logViewModel
        ..loadLogs()
        ..loadConversation()
        ..startStreaming();
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return DefaultTabController(
      length: 5,
      child: Column(
        children: [
          Container(
            decoration: BoxDecoration(
              color: theme.colorScheme.surface,
              boxShadow: [
                BoxShadow(
                  color: theme.colorScheme.shadow.withOpacity(0.06),
                  blurRadius: 12,
                  offset: const Offset(0, 4),
                ),
              ],
            ),
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
            child: Align(
              alignment: Alignment.centerLeft,
              child: DecoratedBox(
                decoration: BoxDecoration(
                  color: theme.colorScheme.surfaceVariant.withOpacity(0.6),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: TabBar(
                  isScrollable: true,
                  labelStyle: theme.textTheme.titleMedium,
                  indicator: BoxDecoration(
                    color: theme.colorScheme.primaryContainer,
                    borderRadius: BorderRadius.circular(14),
                  ),
                  labelColor: theme.colorScheme.onPrimaryContainer,
                  unselectedLabelColor: theme.colorScheme.onSurfaceVariant,
                  tabs: const [
                    Tab(icon: Icon(Icons.library_books), text: 'Knowledge'),
                    Tab(icon: Icon(Icons.memory), text: 'Memory'),
                    Tab(icon: Icon(Icons.trending_up), text: 'Self-Learning'),
                    Tab(icon: Icon(Icons.smart_toy), text: 'Agents'),
                    Tab(icon: Icon(Icons.history), text: 'Logs'),
                  ],
                ),
              ),
            ),
          ),
          Expanded(
            child: Container(
              color: theme.colorScheme.background,
              child: TabBarView(
                children: [
                  ChangeNotifierProvider.value(
                    value: widget.knowledgeViewModel,
                    child: const KnowledgeBaseSection(),
                  ),
                  ChangeNotifierProvider.value(
                    value: widget.memoryViewModel,
                    child: const MemoryManagementSection(),
                  ),
                  ChangeNotifierProvider.value(
                    value: widget.learningViewModel,
                    child: const LearningProgressSection(),
                  ),
                  ChangeNotifierProvider.value(
                    value: widget.agentStatusViewModel,
                    child: const AgentStatusSection(),
                  ),
                  ChangeNotifierProvider.value(
                    value: widget.logViewModel,
                    child: const LogsSection(),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
