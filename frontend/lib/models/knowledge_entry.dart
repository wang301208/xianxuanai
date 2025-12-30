import 'package:collection/collection.dart';

/// Representation of an individual knowledge entry exposed to the UI.
///
/// The model mirrors the structure returned by the backend knowledge APIs and
/// is intentionally immutable to keep state handling predictable inside the
/// view models.
class KnowledgeEntry {
  KnowledgeEntry({
    required this.id,
    required this.title,
    required this.summary,
    required this.content,
    required this.tags,
    required this.source,
    required this.updatedAt,
  });

  final String id;
  final String title;
  final String summary;
  final String content;
  final List<String> tags;
  final String source;
  final DateTime updatedAt;

  factory KnowledgeEntry.fromMap(Map<String, dynamic> map) {
    return KnowledgeEntry(
      id: map['id'] as String,
      title: map['title'] as String? ?? '',
      summary: map['summary'] as String? ?? '',
      content: map['content'] as String? ?? '',
      tags: (map['tags'] as List<dynamic>? ?? const [])
          .map((dynamic value) => value.toString())
          .toList(growable: false),
      source: map['source'] as String? ?? 'unknown',
      updatedAt: DateTime.tryParse(map['updatedAt'] as String? ?? '') ??
          (map['updated_at'] is int
              ? DateTime.fromMillisecondsSinceEpoch(
                  map['updated_at'] as int,
                  isUtc: true,
                ).toLocal()
              : DateTime.now()),
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'title': title,
      'summary': summary,
      'content': content,
      'tags': tags,
      'source': source,
      'updatedAt': updatedAt.toIso8601String(),
    };
  }

  KnowledgeEntry copyWith({
    String? title,
    String? summary,
    String? content,
    List<String>? tags,
    String? source,
    DateTime? updatedAt,
  }) {
    return KnowledgeEntry(
      id: id,
      title: title ?? this.title,
      summary: summary ?? this.summary,
      content: content ?? this.content,
      tags: tags ?? this.tags,
      source: source ?? this.source,
      updatedAt: updatedAt ?? this.updatedAt,
    );
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        other is KnowledgeEntry &&
            other.id == id &&
            other.title == title &&
            other.summary == summary &&
            other.content == content &&
            const ListEquality<String>().equals(other.tags, tags) &&
            other.source == source &&
            other.updatedAt == updatedAt;
  }

  @override
  int get hashCode => Object.hash(
        id,
        title,
        summary,
        content,
        Object.hashAll(tags),
        source,
        updatedAt,
      );

  @override
  String toString() {
    return 'KnowledgeEntry(id: $id, title: $title, tags: $tags)';
  }
}

/// Payload used when creating or editing a knowledge entry.
class KnowledgeEntryDraft {
  KnowledgeEntryDraft({
    required this.title,
    required this.summary,
    required this.content,
    List<String>? tags,
    this.source,
  }) : tags = List.unmodifiable(tags ?? const []);

  final String title;
  final String summary;
  final String content;
  final List<String> tags;
  final String? source;

  Map<String, dynamic> toJson() {
    return {
      'title': title,
      'summary': summary,
      'content': content,
      'tags': tags,
      if (source != null) 'source': source,
    };
  }
}
