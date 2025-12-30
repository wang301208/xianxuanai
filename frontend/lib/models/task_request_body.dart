class TaskRequestBody {
  final String input;
  final Map<String, dynamic>? additionalInput;

  TaskRequestBody({required this.input, this.additionalInput});

  Map<String, dynamic> toJson() => {
        'input': input,
        if (additionalInput != null) 'additional_input': additionalInput,
      };
}
