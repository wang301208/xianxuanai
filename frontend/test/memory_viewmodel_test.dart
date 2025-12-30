import 'package:auto_gpt_flutter_client/models/memory_entry.dart';
import 'package:auto_gpt_flutter_client/services/memory_service.dart';
import 'package:auto_gpt_flutter_client/viewmodels/memory_viewmodel.dart';
import 'package:flutter_test/flutter_test.dart';

class _FakeMemoryService implements MemoryService {
  _FakeMemoryService()
      : _entries = [
          MemoryEntry(
            id: 'a',
            text: 'Remember to review pull requests',
            source: 'workflow',
            similarity: 0.92,
            createdAt: DateTime.utc(2024, 1, 1),
            lastAccessed: DateTime.utc(2024, 2, 1),
            usage: 4,
          ),
          MemoryEntry(
            id: 'b',
            text: 'Data ingestion completed for dataset X',
            source: 'pipeline',
            similarity: 0.81,
            createdAt: DateTime.utc(2024, 1, 2),
            lastAccessed: DateTime.utc(2024, 2, 2),
            usage: 2,
          ),
        ];

  List<MemoryEntry> _entries;

  @override
  Future<void> clearAll() async {
    _entries = [];
  }

  @override
  Future<void> deleteEntry(String id) async {
    _entries.removeWhere((entry) => entry.id == id);
  }

  @override
  Future<List<MemoryEntry>> fetchRecent({int? limit}) async {
    if (limit != null) {
      return _entries.take(limit).toList();
    }
    return List<MemoryEntry>.from(_entries);
  }

  @override
  Future<List<MemoryEntry>> search(String query, {int? limit}) async {
    final lower = query.toLowerCase();
    final result = _entries
        .where((entry) => entry.text.toLowerCase().contains(lower))
        .toList();
    if (limit == null) {
      return result;
    }
    return result.take(limit).toList();
  }

  @override
  Future<Map<String, int>> stats() async {
    return {
      'total_entries': _entries.length,
      'promoted': _entries.where((entry) => entry.promoted).length,
    };
  }
}

void main() {
  late MemoryViewModel viewModel;
  late _FakeMemoryService service;

  setUp(() {
    service = _FakeMemoryService();
    viewModel = MemoryViewModel(service: service);
  });

  test('loadRecent populates entries and stats', () async {
    await viewModel.loadRecent();

    expect(viewModel.entries, hasLength(2));
    expect(viewModel.stats['total_entries'], 2);
  });

  test('search filters entries and sets search term', () async {
    await viewModel.loadRecent();

    await viewModel.search('dataset');

    expect(viewModel.entries, hasLength(1));
    expect(viewModel.entries.first.id, 'b');
    expect(viewModel.hasActiveSearch, isTrue);
  });

  test('clearEntry removes a specific entry', () async {
    await viewModel.loadRecent();

    await viewModel.clearEntry('a');

    expect(viewModel.entries, hasLength(1));
    expect(viewModel.entries.first.id, 'b');
  });

  test('clearAll removes all entries', () async {
    await viewModel.loadRecent();

    await viewModel.clearAll();

    expect(viewModel.entries, isEmpty);
    expect(viewModel.stats['total_entries'], 0);
  });
}
