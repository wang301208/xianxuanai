/// Representation of a single short-term memory or embedding record.
class MemoryEntry {
  MemoryEntry({
    required this.id,
    required this.text,
    required this.source,
    this.similarity,
    this.metadata = const {},
    this.createdAt,
    this.lastAccessed,
    this.usage = 0,
    this.promoted = false,
  });

  final String id;
  final String text;
  final String source;
  final double? similarity;
  final Map<String, dynamic> metadata;
  final DateTime? createdAt;
  final DateTime? lastAccessed;
  final int usage;
  final bool promoted;

  factory MemoryEntry.fromMap(Map<String, dynamic> map) {
    return MemoryEntry(
      id: map['id']?.toString() ?? '',
      text: map['text'] as String? ?? '',
      source: map['source'] as String? ?? 'unknown',
      similarity: map['similarity'] is num
          ? (map['similarity'] as num).toDouble()
          : null,
      metadata: Map<String, dynamic>.from(
        map['metadata'] as Map? ?? const {},
      ),
      createdAt: _timestampToDate(map['createdAt'] ?? map['created_at']),
      lastAccessed: _timestampToDate(map['lastAccessed'] ?? map['last_accessed']),
      usage: map['usage'] is int ? map['usage'] as int : int.tryParse('${map['usage']}') ?? 0,
      promoted: map['promoted'] == true,
    );
  }

  static DateTime? _timestampToDate(dynamic value) {
    if (value == null) {
      return null;
    }
    if (value is String) {
      return DateTime.tryParse(value);
    }
    if (value is int) {
      return DateTime.fromMillisecondsSinceEpoch(value, isUtc: true).toLocal();
    }
    if (value is double) {
      return DateTime.fromMillisecondsSinceEpoch(value.toInt(), isUtc: true).toLocal();
    }
    return null;
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'text': text,
      'source': source,
      'similarity': similarity,
      'metadata': metadata,
      'createdAt': createdAt?.toIso8601String(),
      'lastAccessed': lastAccessed?.toIso8601String(),
      'usage': usage,
      'promoted': promoted,
    };
  }
}
