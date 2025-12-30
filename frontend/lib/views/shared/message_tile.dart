import 'package:flutter/material.dart';

/// A simple tile that shows a message with a title label.
/// This is used by [UserMessageTile] and [AgentMessageTile]
/// to ensure consistent styling.
class MessageTile extends StatelessWidget {
  final String title;
  final String message;
  final Widget? trailing;

  const MessageTile({
    super.key,
    required this.title,
    required this.message,
    this.trailing,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(title, style: Theme.of(context).textTheme.labelMedium),
        Text(
          message,
          softWrap: true,
        ),
        if (trailing != null) trailing!,
      ],
    );
  }
}
