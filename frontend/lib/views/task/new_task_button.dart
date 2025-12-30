import 'package:flutter/material.dart';

import '../shared/app_button.dart';

class NewTaskButton extends StatelessWidget {
  final VoidCallback onPressed;
  const NewTaskButton({super.key, required this.onPressed});

  @override
  Widget build(BuildContext context) {
    return PrimaryButton(label: 'New Task', onPressed: onPressed);
  }
}
