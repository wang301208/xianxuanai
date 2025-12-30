import 'dart:async';
import 'dart:convert';

import 'package:auto_gpt_flutter_client/viewmodels/collaboration_viewmodel.dart';
import 'package:auto_gpt_flutter_client/views/collaboration/collaboration_panel.dart';
import 'package:auto_gpt_flutter_client/views/shared/app_button.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

class FakeWebSocketSink implements WebSocketSink {
  final StreamController<dynamic> _outgoing;

  FakeWebSocketSink(this._outgoing);

  @override
  void add(event) => _outgoing.add(event);

  @override
  void addError(error, [StackTrace? stackTrace]) => _outgoing.addError(error, stackTrace);

  @override
  Future close([int? closeCode, String? closeReason]) => _outgoing.close();

  @override
  Future get done => _outgoing.done;
}

class FakeWebSocketChannel implements WebSocketChannel {
  final StreamController<dynamic> incoming;
  final StreamController<dynamic> outgoing;

  FakeWebSocketChannel(this.incoming, this.outgoing);

  @override
  Stream get stream => incoming.stream;

  @override
  WebSocketSink get sink => FakeWebSocketSink(outgoing);
}

void main() {
  testWidgets('shows server data and sends user input', (tester) async {
    final incoming = StreamController<dynamic>();
    final outgoing = StreamController<dynamic>();
    final channel = FakeWebSocketChannel(incoming, outgoing);
    final viewModel = CollaborationViewModel(channel);

    await tester.pumpWidget(MaterialApp(home: CollaborationPanel(viewModel: viewModel)));

    incoming.add(jsonEncode({'plan': 'P1', 'worldModel': 'WM', 'metrics': 'M'}));
    await tester.pump();

    expect(find.text('Plan: P1'), findsOneWidget);
    expect(find.text('World Model: WM'), findsOneWidget);
    expect(find.text('Metrics: M'), findsOneWidget);
    // Two primary buttons for Inject and Correct
    expect(find.byType(PrimaryButton), findsNWidgets(2));

    await tester.enterText(find.byType(TextField), 'hello');
    await tester.tap(find.text('Inject'));
    await tester.pump();

    final sent = await outgoing.stream.first as String;
    expect(jsonDecode(sent)['content'], 'hello');
  });
}
