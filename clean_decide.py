import pathlib
import textwrap

path = pathlib.Path('modules/brain/whole_brain.py')
text = path.read_text(encoding='utf-8')
start = text.find('    def decide(')
if start == -1:
    raise SystemExit('decide start not found')
end = text.find('@dataclass', start)
if end == -1:
    raise SystemExit('end marker not found')
block = textwrap.dedent('''
    def decide(
        self,
        perception: PerceptionSnapshot,
        emotion: EmotionSnapshot,
        personality: PersonalityProfile,
        curiosity: CuriosityState,
        learning_prediction: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = context or {}
        summary = self._summarise_perception(perception)
        focus = max(summary, key=summary.get) if summary else None
        options = {
            "observe": 0.2 + (1 - abs(emotion.dimensions.get("valence", 0.0))) * 0.3,
            "approach": 0.2 + emotion.intent_bias.get("approach", 0.0),
            "withdraw": 0.2 + emotion.intent_bias.get("withdraw", 0.0),
            "explore": 0.2 + emotion.intent_bias.get("explore", 0.0) + curiosity.drive * 0.5,
        }
        if learning_prediction:
            predicted_load = float(learning_prediction.get("cpu", 0.0))
            resource_pressure = float(learning_prediction.get("memory", 0.0))
            options["observe"] += max(0.0, predicted_load - 0.5) * 0.3
            options["withdraw"] += max(0.0, resource_pressure - 0.5) * 0.2
            options["approach"] += max(0.0, 0.5 - predicted_load) * 0.2
        if context.get("threat", 0.0) > 0.4:
            options["withdraw"] += 0.3
        if context.get("safety", 0.0) > 0.5:
            options["approach"] += 0.2
        options["explore"] *= 0.5 + personality.modulation_weight("explore")
        options["approach"] *= 0.5 + personality.modulation_weight("social")
        options["withdraw"] *= 0.5 + personality.modulation_weight("caution")
        options["observe"] *= 0.5 + personality.modulation_weight("persist")
        total = sum(options.values()) or 1.0
        weights = {k: v / total for k, v in options.items()}
        intention = max(weights.items(), key=lambda item: item[1])[0]
        confidence = weights[intention]
        plan = self._build_plan(intention, summary, context, focus)
        tags = [intention]
        if confidence >= 0.65:
            tags.append("high-confidence")
        if curiosity.last_novelty > 0.6:
            tags.append("novelty-driven")
        if focus:
            tags.append(f"focus-{focus}")
        self._remember(summary, emotion, intention, confidence)
        thought_trace = [
            f"focus={focus or 'none'}",
            f"intention={intention}",
            f"emotion={emotion.primary.value}:{emotion.intensity:.2f}",
            f"curiosity={curiosity.drive:.2f}",
        ]
        if learning_prediction:
            thought_trace.append(
                f"predicted_cpu={float(learning_prediction.get('cpu', 0.0)):.2f}"
            )
            thought_trace.append(
                f"predicted_mem={float(learning_prediction.get('memory', 0.0)):.2f}"
            )
        summary_text = ', '.join(f"{k}:{v:.2f}" for k, v in summary.items()) or 'no-salient-modalities'
        decision = {
            "intention": intention,
            "plan": plan,
            "confidence": confidence,
            "weights": weights,
            "tags": tags,
            "focus": focus or intention,
            "summary": summary_text,
            "thought_trace": thought_trace,
            "perception_summary": summary,
        }
        return decision
''').strip('\n')
indented = textwrap.indent(block, '    ') + '\r\n\r\n'
text = text[:start] + indented + text[end:]
path.write_text(text, encoding='utf-8')
