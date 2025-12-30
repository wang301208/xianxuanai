import '../models/memory_entry.dart';

import 'api_client.dart';

abstract class MemoryService {
  Future<List<MemoryEntry>> fetchRecent({int? limit});
  Future<List<MemoryEntry>> search(String query, {int? limit});
  Future<void> deleteEntry(String id);
  Future<void> clearAll();
  Future<Map<String, int>> stats();
}

class HttpMemoryService implements MemoryService {
  HttpMemoryService({
    ApiClient? apiClient,
    this.basePath = '/api/memory/entries',
  }) : _client = apiClient ?? ApiClient();

  final ApiClient _client;
  final String basePath;

  @override
  Future<List<MemoryEntry>> fetchRecent({int? limit}) async {
    final data = await _client.getJsonList(
      basePath,
      query: {
        'sort': 'recent',
        if (limit != null) 'limit': limit,
      },
    );
    return _decodeList(data);
  }

  @override
  Future<List<MemoryEntry>> search(String query, {int? limit}) async {
    final data = await _client.getJsonList(
      basePath,
      query: {
        'search': query,
        if (limit != null) 'limit': limit,
      },
    );
    return _decodeList(data);
  }

  @override
  Future<void> deleteEntry(String id) {
    return _client.delete('$basePath/$id');
  }

  @override
  Future<void> clearAll() {
    return _client.delete(basePath);
  }

  @override
  Future<Map<String, int>> stats() async {
    final data = await _client.getJson('$basePath/stats');
    return data.map(
      (key, value) => MapEntry(
        key,
        value is int ? value : int.tryParse('$value') ?? 0,
      ),
    );
  }

  List<MemoryEntry> _decodeList(List<dynamic> payload) {
    return payload
        .whereType<Map<String, dynamic>>()
        .map(MemoryEntry.fromMap)
        .toList(growable: false);
  }
}
