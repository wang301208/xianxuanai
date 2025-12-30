import 'package:flutter/material.dart';

import '../../models/task.dart';

class TaskListTile extends StatefulWidget {
  final Task task;
  final VoidCallback onTap;
  final VoidCallback onDelete;
  const TaskListTile({
    super.key,
    required this.task,
    required this.onTap,
    required this.onDelete,
  });

  @override
  State<TaskListTile> createState() => _TaskListTileState();
}

class _TaskListTileState extends State<TaskListTile> {
  bool isSelected = false;

  @override
  Widget build(BuildContext context) {
    return ListTile(
      title: Text(widget.task.title),
      onTap: () {
        setState(() {
          isSelected = !isSelected;
        });
        widget.onTap();
      },
      trailing: isSelected
          ? IconButton(
              icon: const Icon(Icons.close),
              onPressed: widget.onDelete,
            )
          : null,
    );
  }
}
