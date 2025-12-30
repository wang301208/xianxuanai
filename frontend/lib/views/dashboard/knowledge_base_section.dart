import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import '../../models/knowledge_entry.dart';
import '../../viewmodels/knowledge_base_viewmodel.dart';
import '../shared/app_button.dart';
import '../shared/empty_state.dart';
import '../shared/panel_container.dart';
import '../shared/section_header.dart';

class KnowledgeBaseSection extends StatefulWidget {
  const KnowledgeBaseSection({super.key});

  @override
  State<KnowledgeBaseSection> createState() => _KnowledgeBaseSectionState();
}

class _KnowledgeBaseSectionState extends State<KnowledgeBaseSection> {
  late final TextEditingController _searchController;

  @override
  void initState() {
    super.initState();
    _searchController = TextEditingController();
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final vm = context.read<KnowledgeBaseViewModel>();
    if (_searchController.text != vm.searchTerm) {
      _searchController.text = vm.searchTerm;
    }
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<KnowledgeBaseViewModel>(
      builder: (context, viewModel, _) {
        final theme = Theme.of(context);
        return Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              SectionHeader(
                icon: Icons.library_books,
                title: 'Knowledge Base',
                subtitle: 'Curate what the agent knows and keep sources aligned.',
                actions: [
                  IconButton(
                    tooltip: 'Refresh entries',
                    icon: const Icon(Icons.refresh),
                    onPressed: viewModel.refresh,
                  ),
                ],
              ),
              const SizedBox(height: 16),
              _FilterRow(
                controller: _searchController,
                onSearch: (value) {
                  viewModel.updateSearchTerm(value);
                  viewModel.loadEntries(query: value);
                },
                onClear: () {
                  _searchController.clear();
                  viewModel.updateSearchTerm('');
                  viewModel.loadEntries(query: '');
                },
                onCreate: () => _showEntryEditor(context),
                hasActiveQuery: viewModel.searchTerm.isNotEmpty,
              ),
              if (viewModel.isLoading) ...[
                const SizedBox(height: 16),
                const LinearProgressIndicator(),
              ],
              if (viewModel.errorMessage != null)
                Padding(
                  padding: const EdgeInsets.only(top: 16),
                  child: MaterialBanner(
                    backgroundColor: theme.colorScheme.errorContainer,
                    content: Text(
                      viewModel.errorMessage!,
                      style: theme.textTheme.bodyMedium?.copyWith(
                        color: theme.colorScheme.onErrorContainer,
                      ),
                    ),
                    actions: [
                      TextButton(
                        onPressed: viewModel.refresh,
                        child: const Text('Retry'),
                      ),
                    ],
                  ),
                ),
              const SizedBox(height: 24),
              Expanded(
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Expanded(
                      flex: 2,
                      child: PanelContainer(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            Text(
                              'Entries',
                              style: theme.textTheme.titleMedium?.copyWith(
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                            const SizedBox(height: 12),
                            Expanded(
                              child: viewModel.entries.isEmpty
                                  ? EmptyState(
                                      icon: Icons.menu_book,
                                      title: 'No knowledge captured yet',
                                      message:
                                          'Add curated facts, policies, or documents so the agent can ground its responses.',
                                      action: PrimaryButton(
                                        label: 'New knowledge entry',
                                        icon: Icons.add,
                                        onPressed: () => _showEntryEditor(context),
                                      ),
                                    )
                                  : ListView.separated(
                                      itemCount: viewModel.entries.length,
                                      separatorBuilder: (_, __) => const Divider(height: 1),
                                      itemBuilder: (context, index) {
                                        final entry = viewModel.entries[index];
                                        final bool selected = viewModel.selectedEntry?.id == entry.id;
                                        return ListTile(
                                          leading: Icon(
                                            Icons.description_outlined,
                                            color: selected
                                                ? theme.colorScheme.primary
                                                : theme.colorScheme.onSurfaceVariant,
                                          ),
                                          title: Text(entry.title),
                                          subtitle: Text(
                                            entry.summary.isEmpty ? 'No summary provided' : entry.summary,
                                            maxLines: 2,
                                            overflow: TextOverflow.ellipsis,
                                          ),
                                          trailing: Column(
                                            mainAxisAlignment: MainAxisAlignment.center,
                                            crossAxisAlignment: CrossAxisAlignment.end,
                                            children: [
                                              Text(
                                                _formatTimestamp(entry.updatedAt),
                                                style: theme.textTheme.bodySmall,
                                              ),
                                              if (entry.tags.isNotEmpty)
                                                Wrap(
                                                  spacing: 4,
                                                  children: entry.tags
                                                      .take(3)
                                                      .map(
                                                        (tag) => Chip(
                                                          label: Text(tag),
                                                          visualDensity: VisualDensity.compact,
                                                          materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                                                        ),
                                                      )
                                                      .toList(growable: false),
                                                ),
                                            ],
                                          ),
                                          selected: selected,
                                          selectedTileColor:
                                              theme.colorScheme.primaryContainer.withOpacity(0.35),
                                          shape: RoundedRectangleBorder(
                                            borderRadius: BorderRadius.circular(12),
                                          ),
                                          onTap: () => viewModel.selectEntry(entry.id),
                                        );
                                      },
                                    ),
                            ),
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(width: 24),
                    Expanded(
                      flex: 3,
                      child: PanelContainer(
                        child: viewModel.selectedEntry == null
                            ? const EmptyState(
                                icon: Icons.visibility_outlined,
                                title: 'Select a knowledge entry',
                                message:
                                    'Choose an item from the list to inspect its source, summary, and detailed content.',
                              )
                            : _KnowledgeEntryDetail(
                                entry: viewModel.selectedEntry!,
                                isSaving: viewModel.isSaving,
                                onEdit: () => _showEntryEditor(
                                  context,
                                  entry: viewModel.selectedEntry!,
                                ),
                              ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  Future<void> _showEntryEditor(
    BuildContext context, {
    KnowledgeEntry? entry,
  }) async {
    final viewModel = context.read<KnowledgeBaseViewModel>();

    final titleController = TextEditingController(text: entry?.title ?? '');
    final summaryController = TextEditingController(text: entry?.summary ?? '');
    final contentController = TextEditingController(text: entry?.content ?? '');
    final tagsController = TextEditingController(
      text: entry == null ? '' : entry.tags.join(', '),
    );
    final sourceController = TextEditingController(text: entry?.source ?? '');
    final formKey = GlobalKey<FormState>();

    final result = await showDialog<bool>(
      context: context,
      builder: (dialogContext) {
        return Consumer<KnowledgeBaseViewModel>(
          builder: (context, vm, _) {
            return AlertDialog(
              title: Row(
                children: [
                  Icon(entry == null ? Icons.add_circle_outline : Icons.edit_note),
                  const SizedBox(width: 8),
                  Text(entry == null ? 'New knowledge entry' : 'Edit knowledge entry'),
                ],
              ),
              content: SizedBox(
                width: 520,
                child: Form(
                  key: formKey,
                  child: SingleChildScrollView(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        TextFormField(
                          controller: titleController,
                          decoration: const InputDecoration(
                            labelText: 'Title',
                            hintText: 'What is the key idea or artifact?',
                          ),
                          validator: (value) {
                            if (value == null || value.trim().isEmpty) {
                              return 'Title is required';
                            }
                            return null;
                          },
                        ),
                        const SizedBox(height: 12),
                        TextFormField(
                          controller: summaryController,
                          decoration: const InputDecoration(
                            labelText: 'Summary',
                            hintText: 'Short description for quick scanning',
                          ),
                          maxLines: 2,
                        ),
                        const SizedBox(height: 12),
                        TextFormField(
                          controller: contentController,
                          decoration: const InputDecoration(
                            labelText: 'Detailed content',
                            hintText: 'The narrative or instructions the agent should follow',
                          ),
                          maxLines: 8,
                          minLines: 6,
                        ),
                        const SizedBox(height: 12),
                        TextFormField(
                          controller: tagsController,
                          decoration: const InputDecoration(
                            labelText: 'Tags (comma-separated)',
                            hintText: 'e.g. retrieval, compliance, onboarding',
                          ),
                        ),
                        const SizedBox(height: 12),
                        TextFormField(
                          controller: sourceController,
                          decoration: const InputDecoration(
                            labelText: 'Source reference',
                            hintText: 'Add a URL or file reference if helpful',
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(dialogContext).pop(),
                  child: const Text('Cancel'),
                ),
                PrimaryButton(
                  label: entry == null ? 'Create entry' : 'Save changes',
                  icon: entry == null ? Icons.check_circle : Icons.save_outlined,
                  tooltip: entry == null ? 'Confirm and add to knowledge base' : 'Persist changes',
                  onPressed: vm.isSaving
                      ? null
                      : () async {
                          if (!formKey.currentState!.validate()) {
                            return;
                          }

                          final draft = KnowledgeEntryDraft(
                            title: titleController.text.trim(),
                            summary: summaryController.text.trim(),
                            content: contentController.text.trim(),
                            tags: tagsController.text
                                .split(',')
                                .map((tag) => tag.trim())
                                .where((tag) => tag.isNotEmpty)
                                .toList(),
                            source: sourceController.text.trim().isEmpty
                                ? null
                                : sourceController.text.trim(),
                          );

                          if (entry == null) {
                            await vm.createEntry(draft);
                          } else {
                            await vm.updateEntry(entry.id, draft);
                          }

                          if (context.mounted) {
                            Navigator.of(dialogContext).pop(true);
                          }
                        },
                ),
              ],
            );
          },
        );
      },
    );
    if (result == true && mounted) {
      final message = entry == null ? 'Knowledge entry created' : 'Knowledge entry updated';
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message)));
    }
  }

  String _formatTimestamp(DateTime? timestamp) {
    if (timestamp == null) {
      return '';
    }
    final local = timestamp.toLocal();
    final month = local.month.toString().padLeft(2, '0');
    final day = local.day.toString().padLeft(2, '0');
    final hour = local.hour.toString().padLeft(2, '0');
    final minute = local.minute.toString().padLeft(2, '0');
    return '${local.year}-$month-$day $hour:$minute';
  }
}

class _FilterRow extends StatelessWidget {
  const _FilterRow({
    required this.controller,
    required this.onSearch,
    required this.onClear,
    required this.onCreate,
    required this.hasActiveQuery,
  });

  final TextEditingController controller;
  final ValueChanged<String> onSearch;
  final VoidCallback onClear;
  final VoidCallback onCreate;
  final bool hasActiveQuery;

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
              labelText: 'Search knowledge base...',
              hintText: 'Filter by title, summary, or tags',
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
          label: 'New entry',
          icon: Icons.add_circle_outline,
          tooltip: 'Create a new knowledge entry',
          onPressed: onCreate,
        ),
      ],
    );
  }
}

class _KnowledgeEntryDetail extends StatelessWidget {
  const _KnowledgeEntryDetail({
    required this.entry,
    required this.isSaving,
    required this.onEdit,
  });

  final KnowledgeEntry entry;
  final bool isSaving;
  final VoidCallback onEdit;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          children: [
            CircleAvatar(
              radius: 18,
              backgroundColor: theme.colorScheme.secondaryContainer,
              child: Icon(
                Icons.auto_stories,
                color: theme.colorScheme.onSecondaryContainer,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(entry.title, style: theme.textTheme.titleLarge),
                  if (entry.summary.isNotEmpty)
                    Padding(
                      padding: const EdgeInsets.only(top: 4),
                      child: Text(
                        entry.summary,
                        style: theme.textTheme.bodyMedium,
                      ),
                    ),
                ],
              ),
            ),
            if (isSaving)
              const Padding(
                padding: EdgeInsets.only(right: 12),
                child: SizedBox(
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
              ),
            Row(
              children: [
                PrimaryButton(
                  label: 'Edit',
                  icon: Icons.edit,
                  isPrimary: false,
                  tooltip: 'Edit this knowledge entry',
                  onPressed: onEdit,
                ),
                const SizedBox(width: 12),
                PrimaryButton(
                  label: 'Copy',
                  icon: Icons.content_copy,
                  isPrimary: false,
                  tooltip: 'Copy entry content to clipboard',
                  onPressed: () {
                    Clipboard.setData(
                      ClipboardData(
                        text:
                            '${entry.title}\n\n${entry.summary.isEmpty ? '' : '${entry.summary}\n\n'}${entry.content}',
                      ),
                    );
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('Entry copied to clipboard')),
                    );
                  },
                ),
              ],
            ),
          ],
        ),
        const SizedBox(height: 16),
        if (entry.tags.isNotEmpty)
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: entry.tags
                .map(
                  (tag) => Chip(
                    label: Text(tag),
                    visualDensity: VisualDensity.compact,
                  ),
                )
                .toList(growable: false),
          ),
        const SizedBox(height: 16),
        Expanded(
          child: DecoratedBox(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(12),
              color: theme.colorScheme.surfaceVariant.withOpacity(0.4),
            ),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: SingleChildScrollView(
                child: SelectableText(
                  entry.content,
                  style: theme.textTheme.bodyLarge,
                ),
              ),
            ),
          ),
        ),
        const SizedBox(height: 16),
        Row(
          children: [
            const Icon(Icons.link, size: 16),
            const SizedBox(width: 6),
            Expanded(
              child: Text(
                'Source: ${entry.source} - Updated ${_formatTimestamp(entry.updatedAt)}',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ),
          ],
        ),
      ],
    );
  }

  String _formatTimestamp(DateTime? timestamp) {
    if (timestamp == null) {
      return '';
    }
    final local = timestamp.toLocal();
    final month = local.month.toString().padLeft(2, '0');
    final day = local.day.toString().padLeft(2, '0');
    final hour = local.hour.toString().padLeft(2, '0');
    final minute = local.minute.toString().padLeft(2, '0');
    return '${local.year}-$month-$day $hour:$minute';
  }
}
