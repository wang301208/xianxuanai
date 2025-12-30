import 'package:flutter/material.dart';

class ChatInputField extends StatefulWidget {
  final ValueChanged<String> onSendPressed;
  final VoidCallback? onVoicePressed;
  final VoidCallback? onImagePressed;
  const ChatInputField({
    super.key,
    required this.onSendPressed,
    this.onVoicePressed,
    this.onImagePressed,
  });

  @override
  State<ChatInputField> createState() => _ChatInputFieldState();
}

class _ChatInputFieldState extends State<ChatInputField> {
  final TextEditingController _controller = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        IconButton(
          tooltip: 'Voice input (optional)',
          icon: const Icon(Icons.mic_none),
          onPressed: widget.onVoicePressed,
        ),
        IconButton(
          tooltip: 'Send image (optional)',
          icon: const Icon(Icons.image_outlined),
          onPressed: widget.onImagePressed,
        ),
        Expanded(
          child: TextField(
            controller: _controller,
            decoration: const InputDecoration(
              hintText: 'Type a message...',
              border: OutlineInputBorder(),
              isDense: true,
            ),
            onSubmitted: (value) => _send(value),
          ),
        ),
        IconButton(
          icon: const Icon(Icons.send),
          onPressed: () => _send(_controller.text),
        ),
      ],
    );
  }

  void _send(String value) {
    final trimmed = value.trim();
    if (trimmed.isEmpty) {
      return;
    }
    _controller.clear();
    widget.onSendPressed(trimmed);
  }
}
