import 'dart:async';

import 'package:flutter/foundation.dart';

import '../models/learning_job.dart';
import '../services/learning_service.dart';

class LearningProgressViewModel extends ChangeNotifier {
  LearningProgressViewModel({required LearningProgressService service}) : _service = service;

  final LearningProgressService _service;

  final List<LearningJob> jobs = [];
  bool isLoading = false;
  String? errorMessage;
  StreamSubscription<List<LearningJob>>? _subscription;

  Future<void> loadJobs() async {
    isLoading = true;
    errorMessage = null;
    notifyListeners();
    try {
      final fetched = await _service.fetchJobs();
      jobs
        ..clear()
        ..addAll(fetched);
    } catch (e) {
      errorMessage = e.toString();
    } finally {
      isLoading = false;
      notifyListeners();
    }
  }

  void startStreaming({Duration interval = const Duration(seconds: 15)}) {
    _subscription?.cancel();
    _subscription = _service.watchJobs(interval: interval).listen(
      (event) {
        jobs
          ..clear()
          ..addAll(event);
        notifyListeners();
      },
      onError: (Object error) {
        errorMessage = error.toString();
        notifyListeners();
      },
    );
  }

  Future<void> refresh() => loadJobs();

  @override
  void dispose() {
    _subscription?.cancel();
    super.dispose();
  }
}
