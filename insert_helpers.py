import pathlib
import textwrap

path = pathlib.Path('modules/brain/whole_brain.py')
text = path.read_text(encoding='utf-8')
insert_point = text.find('    def process_cycle(')
if insert_point == -1:
    raise SystemExit('process_cycle marker not found')
new_block = textwrap.dedent('''
    def _compose_thought_snapshot(
        self,
        decision: Dict[str, Any],
        memory_refs: List[dict[str, Any]],
    ) -> ThoughtSnapshot:
        plan_steps = list(decision.get("plan", []))
        summary = decision.get("summary") or (
            ', '.join(plan_steps) if plan_steps else decision.get("intention", "")
        )
        return ThoughtSnapshot(
            focus=str(decision.get("focus", decision.get("intention", "unknown"))),
            summary=summary,
            plan=plan_steps,
            confidence=float(decision.get("confidence", 0.5)),
            memory_refs=memory_refs[-3:],
            tags=list(decision.get("tags", [])),
        )

    def _compose_feeling_snapshot(
        self,
        emotion: EmotionSnapshot,
        oscillation_state: Dict[str, float],
        context_features: Dict[str, Any],
    ) -> FeelingSnapshot:
        descriptor = emotion.primary.value.lower()
        valence = float(emotion.dimensions.get("valence", emotion.mood))
        arousal = float(emotion.dimensions.get("arousal", abs(emotion.mood)))
        confidence = max(0.0, min(1.0, 1.0 - float(emotion.decay)))
        context_tags = {
            key
            for key, value in context_features.items()
            if isinstance(value, (int, float)) and value != 0
        }
        for key in oscillation_state:
            context_tags.add(f"osc_{key}")
        return FeelingSnapshot(
            descriptor=descriptor,
            valence=valence,
            arousal=arousal,
            mood=emotion.mood,
            confidence=confidence,
            context_tags=sorted(context_tags),
        )

''').replace('\n', '\r\n')
text = text[:insert_point] + new_block + text[insert_point:]
path.write_text(text, encoding='utf-8')
