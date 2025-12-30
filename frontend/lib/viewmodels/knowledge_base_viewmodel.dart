import 'package:flutter/foundation.dart';

import '../models/knowledge_entry.dart';
import '../services/knowledge_service.dart';

class KnowledgeBaseViewModel extends ChangeNotifier {
  KnowledgeBaseViewModel({required KnowledgeService service}) : _service = service;

  final KnowledgeService _service;

  final List<KnowledgeEntry> entries = [];
  KnowledgeEntry? selectedEntry;
  bool isLoading = false;
  bool isSaving = false;
  String? errorMessage;
  String _searchTerm = '';

  Future<void> loadEntries({String? query}) async {
    isLoading = true;
    errorMessage = null;
    notifyListeners();
    try {
      final fetched = await _service.fetchKnowledge(query: query);
      entries
        ..clear()
        ..addAll(fetched);
      _searchTerm = query ?? '';

      if (selectedEntry != null) {
        final index = entries.indexWhere((entry) => entry.id == selectedEntry!.id);
        if (index != -1) {
          selectedEntry = entries[index];
        } else {
          selectedEntry = null;
        }
      }
    } catch (e) {
      errorMessage = e.toString();
    } finally {
      isLoading = false;
      notifyListeners();
    }
  }

  Future<void> refresh() => loadEntries(query: _searchTerm);

  Future<void> selectEntry(String id) async {
    try {
      if (entries.isEmpty) {
        return;
      }
      final current = entries.firstWhere(
        (entry) => entry.id == id,
        orElse: () => selectedEntry ?? entries.first,
      );
      selectedEntry = current;
      notifyListeners();
      final detailed = await _service.fetchEntry(id);
      final index = entries.indexWhere((entry) => entry.id == id);
      if (index != -1) {
        entries[index] = detailed;
      }
      selectedEntry = detailed;
      notifyListeners();
    } catch (e) {
      errorMessage = e.toString();
      notifyListeners();
    }
  }

  void clearSelection() {
    selectedEntry = null;
    notifyListeners();
  }

  Future<void> createEntry(KnowledgeEntryDraft draft) async {
    isSaving = true;
    errorMessage = null;
    notifyListeners();
    try {
      final created = await _service.createEntry(draft);
      entries.insert(0, created);
      selectedEntry = created;
    } catch (e) {
      errorMessage = e.toString();
    } finally {
      isSaving = false;
      notifyListeners();
    }
  }

  Future<void> updateEntry(String id, KnowledgeEntryDraft draft) async {
    isSaving = true;
    errorMessage = null;
    notifyListeners();
    try {
      final updated = await _service.updateEntry(id, draft);
      final index = entries.indexWhere((entry) => entry.id == id);
      if (index != -1) {
        entries[index] = updated;
      }
      selectedEntry = updated;
    } catch (e) {
      errorMessage = e.toString();
    } finally {
      isSaving = false;
      notifyListeners();
    }
  }

  String get searchTerm => _searchTerm;

  void updateSearchTerm(String term) {
    _searchTerm = term;
    notifyListeners();
  }
}
