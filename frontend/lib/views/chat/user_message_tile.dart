import 'package:flutter/material.dart';

import '../shared/message_tile.dart';

class UserMessageTile extends StatelessWidget {
  final String message;
  const UserMessageTile({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    return MessageTile(title: 'User', message: message);
  }
}
