import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../models/agent_status.dart';
import '../../viewmodels/agent_status_viewmodel.dart';
import '../shared/empty_state.dart';
import '../shared/panel_container.dart';
import '../shared/section_header.dart';

class AgentStatusSection extends StatelessWidget {
  const AgentStatusSection({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<AgentStatusViewModel>(
      builder: (context, viewModel, _) {
        return Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              SectionHeader(
                icon: Icons.smart_toy,
                title: 'Agents',
                subtitle: 'Monitor runtime health, workload, and potential issues at a glance.',
                actions: [
                  if (viewModel.isLoading)
                    const Padding(
                      padding: EdgeInsets.only(right: 4),
                      child: SizedBox(
                        height: 20,
                        width: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      ),
                    ),
                  IconButton(
                    tooltip: 'Refresh agents',
                    icon: const Icon(Icons.refresh),
                    onPressed: viewModel.loadStatuses,
                  ),
                ],
              ),
              if (viewModel.errorMessage != null)
                Padding(
                  padding: const EdgeInsets.only(top: 16),
                  child: Text(
                    viewModel.errorMessage!,
                    style: Theme.of(context)
                        .textTheme
                        .bodyMedium
                        ?.copyWith(color: Theme.of(context).colorScheme.error),
                  ),
                ),
              const SizedBox(height: 16),
              Expanded(
                child: PanelContainer(
                  child: viewModel.agents.isEmpty
                      ? const EmptyState(
                          icon: Icons.smart_toy,
                          title: 'No active agents detected',
                          message: 'Launch an agent or connect to the orchestrator to see live status updates.',
                        )
                      : LayoutBuilder(
                          builder: (context, constraints) {
                            final crossAxisCount = constraints.maxWidth > 900
                                ? 3
                                : constraints.maxWidth > 600
                                    ? 2
                                    : 1;
                            return GridView.builder(
                              gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                                crossAxisCount: crossAxisCount,
                                childAspectRatio: 1.1,
                                crossAxisSpacing: 16,
                                mainAxisSpacing: 16,
                              ),
                              itemCount: viewModel.agents.length,
                              itemBuilder: (context, index) {
                                return _AgentStatusCard(agent: viewModel.agents[index]);
                              },
                            );
                          },
                        ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

class _AgentStatusCard extends StatelessWidget {
  const _AgentStatusCard({required this.agent});

  final AgentStatus agent;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final stateColor = _stateColor(theme, agent.state);
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(18),
        side: BorderSide(color: theme.colorScheme.outlineVariant),
      ),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                CircleAvatar(
                  radius: 18,
                  backgroundColor: stateColor.withOpacity(0.18),
                  child: Icon(_stateIcon(agent.state), color: stateColor),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(agent.name, style: theme.textTheme.titleMedium),
                      const SizedBox(height: 4),
                      Text(
                        agentStateToText(agent.state),
                        style: theme.textTheme.bodySmall?.copyWith(color: stateColor),
                      ),
                    ],
                  ),
                ),
                IconButton(
                  tooltip: 'More details coming soon',
                  icon: const Icon(Icons.open_in_new),
                  onPressed: () {},
                ),
              ],
            ),
            const SizedBox(height: 12),
            if (agent.currentTask != null && agent.currentTask!.isNotEmpty)
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(Icons.task_alt, size: 16, color: theme.colorScheme.onSurfaceVariant),
                  const SizedBox(width: 6),
                  Expanded(
                    child: Text(
                      agent.currentTask!,
                      style: theme.textTheme.bodyMedium,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                ],
              ),
            if (agent.statusMessage != null && agent.statusMessage!.isNotEmpty) ...[
              const SizedBox(height: 8),
              Text(
                agent.statusMessage!,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
                maxLines: 3,
                overflow: TextOverflow.ellipsis,
              ),
            ],
            const Spacer(),
            if (agent.cpuUsage != null) ...[
              Text(
                'CPU ${(agent.cpuUsage! * 100).toStringAsFixed(1)}%',
                style: theme.textTheme.bodySmall,
              ),
              const SizedBox(height: 4),
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: LinearProgressIndicator(
                  value: agent.cpuUsage!.clamp(0.0, 1.0),
                  minHeight: 8,
                  color: theme.colorScheme.primary,
                  backgroundColor: theme.colorScheme.primary.withOpacity(0.15),
                ),
              ),
            ],
            if (agent.memoryUsage != null) ...[
              const SizedBox(height: 12),
              Text(
                'Memory ${(agent.memoryUsage! * 100).toStringAsFixed(1)}%',
                style: theme.textTheme.bodySmall,
              ),
              const SizedBox(height: 4),
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: LinearProgressIndicator(
                  value: agent.memoryUsage!.clamp(0.0, 1.0),
                  minHeight: 8,
                  color: theme.colorScheme.secondary,
                  backgroundColor: theme.colorScheme.secondary.withOpacity(0.15),
                ),
              ),
            ],
            const SizedBox(height: 12),
            Row(
              children: [
                const Icon(Icons.history, size: 16),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    'Updated ${_formatTimestamp(agent.lastUpdated)}',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Color _stateColor(ThemeData theme, AgentLifecycleState state) {
    switch (state) {
      case AgentLifecycleState.running:
        return theme.colorScheme.primary;
      case AgentLifecycleState.waiting:
        return theme.colorScheme.tertiary;
      case AgentLifecycleState.error:
        return theme.colorScheme.error;
      case AgentLifecycleState.paused:
        return theme.colorScheme.secondary;
      case AgentLifecycleState.idle:
        return theme.colorScheme.outline;
    }
  }

  IconData _stateIcon(AgentLifecycleState state) {
    switch (state) {
      case AgentLifecycleState.running:
        return Icons.play_arrow;
      case AgentLifecycleState.waiting:
        return Icons.pause_circle;
      case AgentLifecycleState.error:
        return Icons.error;
      case AgentLifecycleState.paused:
        return Icons.pause;
      case AgentLifecycleState.idle:
        return Icons.stop_circle;
    }
  }

  String _formatTimestamp(DateTime? timestamp) {
    if (timestamp == null) {
      return 'n/a';
    }
    final local = timestamp.toLocal();
    final month = local.month.toString().padLeft(2, '0');
    final day = local.day.toString().padLeft(2, '0');
    final hour = local.hour.toString().padLeft(2, '0');
    final minute = local.minute.toString().padLeft(2, '0');
    return '${local.year}-$month-$day $hour:$minute';
  }
}
