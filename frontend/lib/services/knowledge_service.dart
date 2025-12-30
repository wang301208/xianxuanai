import '../models/knowledge_entry.dart';

import 'api_client.dart';

abstract class KnowledgeService {
  Future<List<KnowledgeEntry>> fetchKnowledge({String? query});
  Future<KnowledgeEntry> fetchEntry(String id);
  Future<KnowledgeEntry> createEntry(KnowledgeEntryDraft draft);
  Future<KnowledgeEntry> updateEntry(String id, KnowledgeEntryDraft draft);
}

class HttpKnowledgeService implements KnowledgeService {
  HttpKnowledgeService({
    ApiClient? apiClient,
    this.basePath = '/api/knowledge/entries',
  }) : _client = apiClient ?? ApiClient();

  final ApiClient _client;
  final String basePath;

  @override
  Future<List<KnowledgeEntry>> fetchKnowledge({String? query}) async {
    final data = await _client.getJsonList(
      basePath,
      query: {
        if (query != null && query.trim().isNotEmpty) 'search': query.trim(),
      },
    );
    return data
        .whereType<Map<String, dynamic>>()
        .map(KnowledgeEntry.fromMap)
        .toList(growable: false);
  }

  @override
  Future<KnowledgeEntry> fetchEntry(String id) async {
    final data = await _client.getJson('$basePath/$id');
    return KnowledgeEntry.fromMap(data);
  }

  @override
  Future<KnowledgeEntry> createEntry(KnowledgeEntryDraft draft) async {
    final data = await _client.postJson(basePath, body: draft.toJson());
    return KnowledgeEntry.fromMap(data);
  }

  @override
  Future<KnowledgeEntry> updateEntry(String id, KnowledgeEntryDraft draft) async {
    final data = await _client.putJson('$basePath/$id', body: draft.toJson());
    return KnowledgeEntry.fromMap(data);
  }
}
