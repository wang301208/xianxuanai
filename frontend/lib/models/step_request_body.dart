class StepRequestBody {
  final String input;
  final Map<String, dynamic>? additionalInput;

  StepRequestBody({required this.input, this.additionalInput});

  Map<String, dynamic> toJson() => {
        'input': input,
        if (additionalInput != null) 'additional_input': additionalInput,
      };
}
