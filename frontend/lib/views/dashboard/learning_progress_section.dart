import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../models/learning_job.dart';
import '../../viewmodels/learning_progress_viewmodel.dart';
import '../shared/empty_state.dart';
import '../shared/panel_container.dart';
import '../shared/section_header.dart';

class LearningProgressSection extends StatelessWidget {
  const LearningProgressSection({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<LearningProgressViewModel>(
      builder: (context, viewModel, _) {
        return Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              SectionHeader(
                icon: Icons.trending_up,
                title: 'Self-learning',
                subtitle: 'Track autonomous fine-tuning, self-evaluation, and knowledge distillation runs.',
                actions: [
                  IconButton(
                    tooltip: 'Refresh learning jobs',
                    icon: const Icon(Icons.refresh),
                    onPressed: viewModel.refresh,
                  ),
                ],
              ),
              const SizedBox(height: 16),
              Expanded(
                child: PanelContainer(
                  child: RefreshIndicator(
                    onRefresh: viewModel.refresh,
                    child: viewModel.isLoading && viewModel.jobs.isEmpty
                        ? const Center(child: CircularProgressIndicator())
                        : viewModel.jobs.isEmpty
                            ? const EmptyState(
                                icon: Icons.self_improvement,
                                title: 'No self-learning jobs yet',
                                message:
                                    'Kick off a training or self-evaluation cycle to monitor progress and metrics here.',
                              )
                            : ListView.separated(
                                physics: const AlwaysScrollableScrollPhysics(),
                                itemCount: viewModel.jobs.length,
                                separatorBuilder: (_, __) => const SizedBox(height: 16),
                                itemBuilder: (context, index) => _LearningJobCard(
                                  job: viewModel.jobs[index],
                                ),
                              ),
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

class _LearningJobCard extends StatelessWidget {
  const _LearningJobCard({required this.job});

  final LearningJob job;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final progressValue = job.totalIterations > 0
        ? (job.iteration / job.totalIterations).clamp(0.0, 1.0)
        : job.progress.clamp(0.0, 1.0);
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
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
                  backgroundColor: theme.colorScheme.primaryContainer,
                  child: Icon(
                    Icons.science,
                    color: theme.colorScheme.onPrimaryContainer,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        job.name,
                        style: theme.textTheme.titleMedium,
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Iteration ${job.iteration} of ${job.totalIterations}',
                        style: theme.textTheme.bodySmall,
                      ),
                    ],
                  ),
                ),
                Chip(
                  avatar: Icon(_statusIcon(job.status), size: 16),
                  label: Text(job.status.toUpperCase()),
                  backgroundColor: _statusColor(theme, job.status),
                  materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                ),
              ],
            ),
            if (job.description != null && job.description!.isNotEmpty) ...[
              const SizedBox(height: 12),
              Text(job.description!, style: theme.textTheme.bodyMedium),
            ],
            const SizedBox(height: 12),
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: LinearProgressIndicator(value: progressValue, minHeight: 10),
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: job.metrics.entries
                  .map(
                    (metric) => Chip(
                      avatar: const Icon(Icons.insights, size: 16),
                      label: Text('${metric.key}: ${metric.value.toStringAsFixed(3)}'),
                      visualDensity: VisualDensity.compact,
                    ),
                  )
                  .toList(growable: false),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                const Icon(Icons.schedule, size: 16),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    'Started ${_formatTimestamp(job.startedAt)} • Updated ${_formatTimestamp(job.updatedAt)}',
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

  Color _statusColor(ThemeData theme, String status) {
    final normalized = status.toLowerCase();
    if (normalized.contains('running') || normalized.contains('active')) {
      return theme.colorScheme.primaryContainer;
    }
    if (normalized.contains('error') || normalized.contains('failed')) {
      return theme.colorScheme.errorContainer;
    }
    if (normalized.contains('completed') || normalized.contains('done')) {
      return theme.colorScheme.secondaryContainer;
    }
    return theme.colorScheme.surfaceVariant;
  }

  IconData _statusIcon(String status) {
    final normalized = status.toLowerCase();
    if (normalized.contains('running') || normalized.contains('active')) {
      return Icons.play_circle;
    }
    if (normalized.contains('error') || normalized.contains('failed')) {
      return Icons.error;
    }
    if (normalized.contains('completed') || normalized.contains('done')) {
      return Icons.check_circle;
    }
    return Icons.timelapse;
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
