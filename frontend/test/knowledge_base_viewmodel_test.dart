import 'package:auto_gpt_flutter_client/models/knowledge_entry.dart';
import 'package:auto_gpt_flutter_client/services/knowledge_service.dart';
import 'package:auto_gpt_flutter_client/viewmodels/knowledge_base_viewmodel.dart';
import 'package:flutter_test/flutter_test.dart';

class _FakeKnowledgeService implements KnowledgeService {
  _FakeKnowledgeService() {
    _entries = [
      KnowledgeEntry(
        id: '1',
        title: 'AI Safety Guidelines',
        summary: 'Key steps to ensure safe deployment.',
        content: 'Always review prompts and monitor outputs.',
        tags: const ['safety', 'deployment'],
        source: 'manual',
        updatedAt: DateTime.utc(2024, 1, 1),
      ),
      KnowledgeEntry(
        id: '2',
        title: 'Data Retention Policy',
        summary: 'How long we keep various datasets.',
        content: 'Short-term data for 30 days, logs for 90 days.',
        tags: const ['policy', 'compliance'],
        source: 'policy.pdf',
        updatedAt: DateTime.utc(2024, 2, 1),
      ),
    ];
  }

  late List<KnowledgeEntry> _entries;
  bool throwOnFetch = false;

  @override
  Future<KnowledgeEntry> createEntry(KnowledgeEntryDraft draft) async {
    final newEntry = KnowledgeEntry(
      id: (_entries.length + 1).toString(),
      title: draft.title,
      summary: draft.summary,
      content: draft.content,
      tags: List<String>.from(draft.tags),
      source: draft.source ?? 'user',
      updatedAt: DateTime.utc(2024, 3, 1),
    );
    _entries.insert(0, newEntry);
    return newEntry;
  }

  @override
  Future<KnowledgeEntry> fetchEntry(String id) async {
    return _entries.firstWhere((entry) => entry.id == id);
  }

  @override
  Future<List<KnowledgeEntry>> fetchKnowledge({String? query}) async {
    if (throwOnFetch) {
      throw Exception('boom');
    }
    if (query == null || query.isEmpty) {
      return List<KnowledgeEntry>.from(_entries);
    }
    final lower = query.toLowerCase();
    return _entries
        .where(
          (entry) =>
              entry.title.toLowerCase().contains(lower) ||
              entry.summary.toLowerCase().contains(lower),
        )
        .toList();
  }

  @override
  Future<KnowledgeEntry> updateEntry(String id, KnowledgeEntryDraft draft) async {
    final index = _entries.indexWhere((entry) => entry.id == id);
    final updated = KnowledgeEntry(
      id: id,
      title: draft.title,
      summary: draft.summary,
      content: draft.content,
      tags: List<String>.from(draft.tags),
      source: draft.source ?? 'user',
      updatedAt: DateTime.utc(2024, 4, 1),
    );
    if (index != -1) {
      _entries[index] = updated;
    }
    return updated;
  }
}

void main() {
  late KnowledgeBaseViewModel viewModel;
  late _FakeKnowledgeService service;

  setUp(() {
    service = _FakeKnowledgeService();
    viewModel = KnowledgeBaseViewModel(service: service);
  });

  test('loadEntries populates entries and clears error', () async {
    expect(viewModel.entries, isEmpty);

    await viewModel.loadEntries();

    expect(viewModel.entries, hasLength(2));
    expect(viewModel.errorMessage, isNull);
    expect(viewModel.isLoading, isFalse);
  });

  test('createEntry inserts a new entry and selects it', () async {
    await viewModel.loadEntries();
    final draft = KnowledgeEntryDraft(
      title: 'New Knowledge',
      summary: 'Summary',
      content: 'Detailed content',
      tags: const ['new'],
      source: 'user',
    );

    await viewModel.createEntry(draft);

    expect(viewModel.entries, hasLength(3));
    expect(viewModel.selectedEntry?.title, 'New Knowledge');
    expect(viewModel.errorMessage, isNull);
  });

  test('updateEntry replaces existing entry', () async {
    await viewModel.loadEntries();
    final draft = KnowledgeEntryDraft(
      title: 'AI Safety Guidelines v2',
      summary: 'Updated summary',
      content: 'Updated content',
      tags: const ['safety'],
      source: 'manual',
    );

    await viewModel.updateEntry('1', draft);

    expect(viewModel.selectedEntry?.title, 'AI Safety Guidelines v2');
    expect(
      viewModel.entries.firstWhere((entry) => entry.id == '1').summary,
      'Updated summary',
    );
  });

  test('loadEntries surfaces service errors', () async {
    service.throwOnFetch = true;

    await viewModel.loadEntries();

    expect(viewModel.errorMessage, isNotNull);
    expect(viewModel.entries, isEmpty);
  });
}
