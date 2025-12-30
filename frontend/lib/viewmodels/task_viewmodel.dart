import 'package:flutter/foundation.dart';

import '../models/task.dart';
import 'mock_data.dart';

class TaskViewModel extends ChangeNotifier {
  List<Task> tasks = [];
  Task? selectedTask;

  void fetchTasks() {
    tasks = List<Task>.from(mockTasks);
    notifyListeners();
  }

  void selectTask(int id) {
    try {
      selectedTask = tasks.firstWhere((task) => task.id == id);
      notifyListeners();
    } catch (_) {
      throw ArgumentError('Task with id $id not found');
    }
  }

  void createTask(String title) {
    final newTask = Task(
        id: tasks.isEmpty ? 1 : tasks.last.id + 1, title: title);
    tasks.add(newTask);
    notifyListeners();
  }

  void deleteTask(int id) {
    final taskExists = tasks.any((task) => task.id == id);
    if (!taskExists) {
      throw ArgumentError('Task with id $id not found');
    }
    tasks.removeWhere((task) => task.id == id);
    notifyListeners();
  }
}
