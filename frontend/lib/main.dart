import 'package:flutter/material.dart';

import 'services/api_client.dart';
import 'services/agent_status_service.dart';
import 'services/knowledge_service.dart';
import 'services/learning_service.dart';
import 'services/log_service.dart';
import 'services/memory_service.dart';
import 'viewmodels/agent_status_viewmodel.dart';
import 'viewmodels/knowledge_base_viewmodel.dart';
import 'viewmodels/learning_progress_viewmodel.dart';
import 'viewmodels/log_viewmodel.dart';
import 'viewmodels/memory_viewmodel.dart';
import 'views/dashboard/system_dashboard.dart';

void main() {
  runApp(const AutoAIDashboardApp());
}

class AutoAIDashboardApp extends StatefulWidget {
  const AutoAIDashboardApp({super.key});

  @override
  State<AutoAIDashboardApp> createState() => _AutoAIDashboardAppState();
}

class _AutoAIDashboardAppState extends State<AutoAIDashboardApp> {
  late final ApiClient _client;

  late final KnowledgeBaseViewModel _knowledgeViewModel;
  late final MemoryViewModel _memoryViewModel;
  late final LearningProgressViewModel _learningViewModel;
  late final AgentStatusViewModel _agentStatusViewModel;
  late final LogViewModel _logViewModel;

  @override
  void initState() {
    super.initState();
    const baseUrl = String.fromEnvironment(
      'AUTOAI_API_BASE_URL',
      defaultValue: 'http://localhost:8000',
    );
    _client = ApiClient(baseUrl: baseUrl);

    _knowledgeViewModel = KnowledgeBaseViewModel(
      service: HttpKnowledgeService(apiClient: _client),
    );
    _memoryViewModel = MemoryViewModel(
      service: HttpMemoryService(apiClient: _client),
    );
    _learningViewModel = LearningProgressViewModel(
      service: HttpLearningProgressService(apiClient: _client),
    );
    _agentStatusViewModel = AgentStatusViewModel(
      service: HttpAgentStatusService(apiClient: _client),
    );
    _logViewModel = LogViewModel(
      service: HttpLogService(apiClient: _client),
    );
  }

  @override
  void dispose() {
    _knowledgeViewModel.dispose();
    _memoryViewModel.dispose();
    _learningViewModel.dispose();
    _agentStatusViewModel.dispose();
    _logViewModel.dispose();
    _client.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AutoAI Dashboard',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: Scaffold(
        body: SafeArea(
          child: SystemDashboard(
            knowledgeViewModel: _knowledgeViewModel,
            memoryViewModel: _memoryViewModel,
            learningViewModel: _learningViewModel,
            agentStatusViewModel: _agentStatusViewModel,
            logViewModel: _logViewModel,
          ),
        ),
      ),
    );
  }
}

