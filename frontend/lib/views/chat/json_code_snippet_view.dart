import 'package:flutter/material.dart';

class JsonCodeSnippetView extends StatelessWidget {
  final String jsonString;
  const JsonCodeSnippetView({super.key, required this.jsonString});

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      child: Text(jsonString),
    );
  }
}
