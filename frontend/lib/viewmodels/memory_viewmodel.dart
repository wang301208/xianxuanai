import 'package:flutter/foundation.dart';

import '../models/memory_entry.dart';
import '../services/memory_service.dart';

class MemoryViewModel extends ChangeNotifier {
  MemoryViewModel({required MemoryService service}) : _service = service;

  final MemoryService _service;

  final List<MemoryEntry> entries = [];
  Map<String, int> stats = const {};
  bool isLoading = false;
  String? errorMessage;
  String _searchTerm = '';

  Future<void> loadRecent() async {
    await _load(() => _service.fetchRecent());
  }

  Future<void> search(String query) async {
    _searchTerm = query.trim();
    if (_searchTerm.isEmpty) {
      await loadRecent();
      return;
    }
    await _load(() => _service.search(_searchTerm));
  }

  Future<void> refreshStats() async {
    try {
      stats = await _service.stats();
      notifyListeners();
    } catch (_) {
      // stats failure is non-fatal
    }
  }

  Future<void> clearEntry(String id) async {
    try {
      await _service.deleteEntry(id);
      entries.removeWhere((entry) => entry.id == id);
      await refreshStats();
      notifyListeners();
    } catch (e) {
      errorMessage = e.toString();
      notifyListeners();
    }
  }

  Future<void> clearAll() async {
    try {
      await _service.clearAll();
      entries.clear();
      await refreshStats();
      notifyListeners();
    } catch (e) {
      errorMessage = e.toString();
      notifyListeners();
    }
  }

  Future<void> _load(Future<List<MemoryEntry>> Function() loader) async {
    isLoading = true;
    errorMessage = null;
    notifyListeners();
    try {
      final fetched = await loader();
      entries
        ..clear()
        ..addAll(fetched);
      await refreshStats();
    } catch (e) {
      errorMessage = e.toString();
    } finally {
      isLoading = false;
      notifyListeners();
    }
  }

  bool get hasActiveSearch => _searchTerm.isNotEmpty;
  String get searchTerm => _searchTerm;
}
