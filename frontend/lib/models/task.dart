class Task {
  final int id;
  String _title;

  Task({required this.id, required String title})
      : assert(title.isNotEmpty),
        _title = title;

  factory Task.fromMap(Map<String, dynamic> map) {
    return Task(id: map['id'] as int, title: map['title'] as String);
  }

  String get title => _title;

  set title(String value) {
    if (value.isEmpty) {
      throw ArgumentError('Title cannot be empty');
    }
    _title = value;
  }

  @override
  bool operator ==(Object other) {
    return other is Task && other.id == id && other.title == title;
  }

  @override
  int get hashCode => Object.hash(id, title);

  @override
  String toString() => 'Task(id: $id, title: $title)';
}
