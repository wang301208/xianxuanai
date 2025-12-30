import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../viewmodels/collaboration_viewmodel.dart';
import '../shared/app_button.dart';

class CollaborationPanel extends StatefulWidget {
  final CollaborationViewModel viewModel;

  const CollaborationPanel({super.key, required this.viewModel});

  @override
  State<CollaborationPanel> createState() => _CollaborationPanelState();
}

class _CollaborationPanelState extends State<CollaborationPanel> {
  final TextEditingController _controller = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider.value(
      value: widget.viewModel,
      child: Consumer<CollaborationViewModel>(
        builder: (context, vm, _) => Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Plan: ${vm.plan}'),
            Text('World Model: ${vm.worldModel}'),
            Text('Metrics: ${vm.metrics}'),
            TextField(controller: _controller),
            Row(
              children: [
                PrimaryButton(
                  label: 'Inject',
                  onPressed: () {
                    vm.sendKnowledge(_controller.text);
                    _controller.clear();
                  },
                ),
                const SizedBox(width: 8),
                PrimaryButton(
                  label: 'Correct',
                  onPressed: () {
                    vm.sendCorrection(_controller.text);
                    _controller.clear();
                  },
                ),
              ],
            )
          ],
        ),
      ),
    );
  }
}
