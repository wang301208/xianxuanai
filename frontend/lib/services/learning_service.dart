import 'dart:async';

import '../models/learning_job.dart';

import 'api_client.dart';

abstract class LearningProgressService {
  Future<List<LearningJob>> fetchJobs();
  Future<LearningJob> fetchJob(String id);
  Stream<List<LearningJob>> watchJobs({Duration interval});
}

class HttpLearningProgressService implements LearningProgressService {
  HttpLearningProgressService({
    ApiClient? apiClient,
    this.basePath = '/api/learning/jobs',
    this.pollInterval = const Duration(seconds: 15),
  }) : _client = apiClient ?? ApiClient();

  final ApiClient _client;
  final String basePath;
  final Duration pollInterval;

  @override
  Future<List<LearningJob>> fetchJobs() async {
    final data = await _client.getJsonList(basePath);
    return data
        .whereType<Map<String, dynamic>>()
        .map(LearningJob.fromMap)
        .toList(growable: false);
  }

  @override
  Future<LearningJob> fetchJob(String id) async {
    final data = await _client.getJson('$basePath/$id');
    return LearningJob.fromMap(data);
  }

  @override
  Stream<List<LearningJob>> watchJobs({Duration interval = const Duration(seconds: 15)}) async* {
    while (true) {
      try {
        yield await fetchJobs();
      } catch (_) {
        // Skip errors in the background stream; UI can show last known data.
      }
      await Future.delayed(interval);
    }
  }
}
