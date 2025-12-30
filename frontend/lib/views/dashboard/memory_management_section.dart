import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import '../../models/memory_entry.dart';
import '../../viewmodels/memory_viewmodel.dart';
import '../shared/app_button.dart';
import '../shared/empty_state.dart';
import '../shared/panel_container.dart';
import '../shared/section_header.dart';

class MemoryManagementSection extends StatefulWidget {
  const MemoryManagementSection({super.key});

  @override
  State<MemoryManagementSection> createState() => _MemoryManagementSectionState();
}

class _MemoryManagementSectionState extends State<MemoryManagementSection> {
  late final TextEditingController _searchController;

  @override
  void initState() {
    super.initState();
    _searchController = TextEditingController();
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<MemoryViewModel>(
      builder: (context, viewModel, _) {
        return Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              SectionHeader(
                icon: Icons.memory,
                title: 'Memory',
                subtitle: 'Inspect and curate the agent’s long-term recall.',
                actions: [
                  IconButton(
                    tooltip: 'Refresh memory entries',
                    icon: const Icon(Icons.refresh),
                    onPressed: viewModel.hasActiveSearch
                        ? () => viewModel.search(viewModel.searchTerm)
                        : viewModel.loadRecent,
                  ),
                ],
              ),
              const SizedBox(height: 16),
              _FilterRow(
                controller: _searchController,
                hasActiveQuery: viewModel.hasActiveSearch,
                onSearch: viewModel.search,
                onClear: () {
                  _searchController.clear();
                  viewModel.search('');
                },
                onClearAll: viewModel.entries.isEmpty
                    ? null
                    : () => _confirmClearAll(context, viewModel),
              ),
              if (viewModel.isLoading) ...[
                const SizedBox(height: 16),
                const LinearProgressIndicator(),
              ],
              const SizedBox(height: 16),
              _MemoryStatsBar(stats: viewModel.stats),
              const SizedBox(height: 16),
              Expanded(
                child: PanelContainer(
                  padding: const EdgeInsets.all(0),
                  child: viewModel.entries.isEmpty
                      ? const EmptyState(
                          icon: Icons.memory_outlined,
                          title: 'No memory entries yet',
                          message:
                              'As the agent operates, retrieved facts and vector embeddings will appear here.',
                        )
                      : ListView.separated(
                          padding: const EdgeInsets.all(20),
                          itemCount: viewModel.entries.length,
                          separatorBuilder: (_, __) => const SizedBox(height: 12),
                          itemBuilder: (context, index) {
                            final entry = viewModel.entries[index];
                            return _MemoryEntryTile(
                              entry: entry,
                              onDelete: () => viewModel.clearEntry(entry.id),
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

  Future<void> _confirmClearAll(BuildContext context, MemoryViewModel viewModel) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (dialogContext) {
        return AlertDialog(
          title: Row(
            children: const [
              Icon(Icons.delete_forever),
              SizedBox(width: 8),
              Text('Clear memory'),
            ],
          ),
          content: const Text(
            'This will remove all cached memories and embeddings. This action cannot be undone. Continue?',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(dialogContext).pop(false),
              child: const Text('Cancel'),
            ),
            PrimaryButton(
              label: 'Clear memory',
              icon: Icons.warning_amber,
              onPressed: () => Navigator.of(dialogContext).pop(true),
            ),
          ],
        );
      },
    );
    if (confirmed == true) {
      await viewModel.clearAll();
      if (!mounted) {
        return;
      }
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('All memory entries cleared')),
      );
    }
  }
}

class _FilterRow extends StatelessWidget {
  const _FilterRow({
    required this.controller,
    required this.hasActiveQuery,
    required this.onSearch,
    required this.onClear,
    required this.onClearAll,
  });

  final TextEditingController controller;
  final bool hasActiveQuery;
  final ValueChanged<String> onSearch;
  final VoidCallback onClear;
  final VoidCallback? onClearAll;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: TextField(
            controller: controller,
            decoration: InputDecoration(
              prefixIcon: const Icon(Icons.search),
              suffixIcon: hasActiveQuery
                  ? IconButton(
                      tooltip: 'Clear search',
                      icon: const Icon(Icons.clear),
                      onPressed: onClear,
                    )
                  : null,
              labelText: 'Search memory...',
              hintText: 'Filter by source or content snippet',
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
            textInputAction: TextInputAction.search,
            onSubmitted: onSearch,
          ),
        ),
        const SizedBox(width: 16),
        PrimaryButton(
          label: 'Clear all',
          icon: Icons.cleaning_services_outlined,
          isPrimary: false,
          tooltip: 'Remove all stored memory entries',
          onPressed: onClearAll,
        ),
      ],
    );
  }
}

class _MemoryStatsBar extends StatelessWidget {
  const _MemoryStatsBar({required this.stats});

  final Map<String, int> stats;

  @override
  Widget build(BuildContext context) {
    if (stats.isEmpty) {
      return const SizedBox.shrink();
    }
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: stats.entries
          .map(
            (entry) => Chip(
              avatar: const Icon(Icons.analytics, size: 16),
              label: Text('${entry.key}: ${entry.value}'),
              visualDensity: VisualDensity.compact,
            ),
          )
          .toList(growable: false),
    );
  }
}

class _MemoryEntryTile extends StatefulWidget {
  const _MemoryEntryTile({
    required this.entry,
    required this.onDelete,
  });

  final MemoryEntry entry;
  final VoidCallback onDelete;

  @override
  State<_MemoryEntryTile> createState() => _MemoryEntryTileState();
}

class _MemoryEntryTileState extends State<_MemoryEntryTile> {
  bool _showDetails = false;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final metadataText = widget.entry.metadata.entries
        .map((entry) => '${entry.key}: ${entry.value}')
        .join('\n');
    return Card(
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(14),
      ),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(Icons.bookmark, color: theme.colorScheme.primary),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    widget.entry.text,
                    style: theme.textTheme.titleMedium,
                  ),
                ),
                IconButton(
                  tooltip: 'Delete memory entry',
                  icon: const Icon(Icons.delete_outline),
                  onPressed: widget.onDelete,
                ),
              ],
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                Chip(
                  avatar: const Icon(Icons.auto_awesome, size: 16),
                  label: Text(widget.entry.source),
                  visualDensity: VisualDensity.compact,
                ),
                if (widget.entry.similarity != null)
                  Chip(
                    avatar: const Icon(Icons.percent, size: 16),
                    label: Text('${(widget.entry.similarity! * 100).toStringAsFixed(1)}% similar'),
                  ),
                Chip(
                  avatar: const Icon(Icons.repeat, size: 16),
                  label: Text('Usage ${widget.entry.usage}'),
                ),
                if (widget.entry.promoted)
                  Chip(
                    avatar: const Icon(Icons.star, size: 16),
                    label: const Text('Promoted'),
                    backgroundColor: theme.colorScheme.primaryContainer,
                  ),
              ],
            ),
            const SizedBox(height: 12),
            Text(
              'Created ${_formatTimestamp(widget.entry.createdAt)} - Last accessed ${_formatTimestamp(widget.entry.lastAccessed)}',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 12,
              runSpacing: 12,
              children: [
                PrimaryButton(
                  label: _showDetails ? 'Hide details' : 'View details',
                  icon: _showDetails ? Icons.expand_less : Icons.expand_more,
                  isPrimary: false,
                  tooltip: 'Toggle memory metadata visibility',
                  onPressed: () {
                    setState(() {
                      _showDetails = !_showDetails;
                    });
                  },
                ),
                PrimaryButton(
                  label: 'Copy text',
                  icon: Icons.copy_outlined,
                  isPrimary: false,
                  tooltip: 'Copy memory text to clipboard',
                  onPressed: () {
                    Clipboard.setData(ClipboardData(text: widget.entry.text));
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('Memory copied to clipboard')),
                    );
                  },
                ),
                if (widget.entry.metadata.isNotEmpty)
                  PrimaryButton(
                    label: 'Copy metadata',
                    icon: Icons.data_object,
                    isPrimary: false,
                    tooltip: 'Copy metadata key-value pairs',
                    onPressed: () {
                      Clipboard.setData(ClipboardData(text: metadataText));
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(content: Text('Metadata copied to clipboard')),
                      );
                    },
                  ),
              ],
            ),
            AnimatedCrossFade(
              firstChild: const SizedBox.shrink(),
              secondChild: Padding(
                padding: const EdgeInsets.only(top: 12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    if (widget.entry.metadata.isNotEmpty)
                      Text(
                        metadataText,
                        style: theme.textTheme.bodySmall,
                      ),
                    if (widget.entry.metadata.isEmpty)
                      Text(
                        'No metadata recorded for this memory.',
                        style: theme.textTheme.bodySmall,
                      ),
                  ],
                ),
              ),
              crossFadeState:
                  _showDetails ? CrossFadeState.showSecond : CrossFadeState.showFirst,
              duration: const Duration(milliseconds: 200),
            ),
          ],
        ),
      ),
    );
  }

  String _formatTimestamp(DateTime? timestamp) {
    if (timestamp == null) {
      return 'never';
    }
    final local = timestamp.toLocal();
    final month = local.month.toString().padLeft(2, '0');
    final day = local.day.toString().padLeft(2, '0');
    final hour = local.hour.toString().padLeft(2, '0');
    final minute = local.minute.toString().padLeft(2, '0');
    return '${local.year}-$month-$day $hour:$minute';
  }
}
