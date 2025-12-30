import 'package:flutter/material.dart';

import '../shared/message_tile.dart';
import 'json_code_snippet_view.dart';

class AgentMessageTile extends StatefulWidget {
  final String message;
  const AgentMessageTile({super.key, required this.message});

  @override
  State<AgentMessageTile> createState() => _AgentMessageTileState();
}

class _AgentMessageTileState extends State<AgentMessageTile> {
  bool expanded = false;

  @override
  Widget build(BuildContext context) {
    return MessageTile(
      title: 'Agent',
      message: widget.message,
      trailing: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          IconButton(
            icon: Icon(expanded ? Icons.keyboard_arrow_up : Icons.keyboard_arrow_down),
            onPressed: () {
              setState(() {
                expanded = !expanded;
              });
            },
          ),
          if (expanded) const JsonCodeSnippetView(jsonString: '{}'),
        ],
      ),
    );
  }
}
