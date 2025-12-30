import 'dart:convert';

import 'package:http/http.dart' as http;

class ApiException implements Exception {
  ApiException(this.statusCode, this.message, [this.body]);

  final int statusCode;
  final String message;
  final String? body;

  @override
  String toString() => 'ApiException($statusCode, $message)';
}

class ApiClient {
  ApiClient({
    http.Client? httpClient,
    String baseUrl = 'http://localhost:8000',
  })  : _client = httpClient ?? http.Client(),
        _baseUri = Uri.parse(baseUrl.endsWith('/') ? baseUrl : '$baseUrl/');

  final http.Client _client;
  final Uri _baseUri;

  Uri get baseUri => _baseUri;

  Future<Map<String, dynamic>> getJson(
    String path, {
    Map<String, dynamic>? query,
  }) async {
    final response = await _client.get(_resolve(path, query));
    _throwForStatus(response);
    return _decodeMap(response);
  }

  Future<List<dynamic>> getJsonList(
    String path, {
    Map<String, dynamic>? query,
  }) async {
    final response = await _client.get(_resolve(path, query));
    _throwForStatus(response);
    return _decodeList(response);
  }

  Future<Map<String, dynamic>> postJson(
    String path, {
    required Map<String, dynamic> body,
  }) async {
    final response = await _client.post(
      _resolve(path),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(body),
    );
    _throwForStatus(response);
    return _decodeMap(response);
  }

  Future<Map<String, dynamic>> putJson(
    String path, {
    required Map<String, dynamic> body,
  }) async {
    final response = await _client.put(
      _resolve(path),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(body),
    );
    _throwForStatus(response);
    return _decodeMap(response);
  }

  Future<void> delete(
    String path, {
    Map<String, dynamic>? query,
  }) async {
    final response = await _client.delete(_resolve(path, query));
    _throwForStatus(response);
  }

  Uri _resolve(String path, [Map<String, dynamic>? query]) {
    final normalizedPath = path.startsWith('/') ? path : '/$path';
    final uri = _baseUri.resolve(normalizedPath);
    if (query == null || query.isEmpty) {
      return uri;
    }
    final filtered = <String, String>{};
    query.forEach((key, value) {
      if (value != null) {
        filtered[key] = value.toString();
      }
    });
    return uri.replace(queryParameters: filtered);
  }

  Map<String, dynamic> _decodeMap(http.Response response) {
    final decoded = jsonDecode(response.body);
    if (decoded is Map<String, dynamic>) {
      return decoded;
    }
    throw ApiException(
      response.statusCode,
      'Expected JSON object',
      response.body,
    );
  }

  List<dynamic> _decodeList(http.Response response) {
    final decoded = jsonDecode(response.body);
    if (decoded is List<dynamic>) {
      return decoded;
    }
    throw ApiException(
      response.statusCode,
      'Expected JSON array',
      response.body,
    );
  }

  void _throwForStatus(http.Response response) {
    if (response.statusCode < 400) {
      return;
    }
    throw ApiException(
      response.statusCode,
      response.reasonPhrase ?? 'Request failed',
      response.body,
    );
  }

  void close() {
    _client.close();
  }
}
