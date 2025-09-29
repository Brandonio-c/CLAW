"""Prompt building and templates for LLM interaction."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from claw.types import Plan


@dataclass
class PromptTemplate:
    """A prompt template with placeholders."""
    system: str
    user: str
    examples: List[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.examples is None:
            self.examples = []


class PromptBuilder:
    """Builder for structured prompts with examples and schemas."""
    
    def __init__(self, domain: str = "diplomacy"):
        """Initialize prompt builder for a specific domain.
        
        Args:
            domain: Domain name (e.g., "diplomacy")
        """
        self.domain = domain
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, PromptTemplate]:
        """Load prompt templates for the domain."""
        if self.domain == "diplomacy":
            return self._load_diplomacy_templates()
        else:
            raise ValueError(f"Unknown domain: {self.domain}")
    
    def _load_diplomacy_templates(self) -> Dict[str, PromptTemplate]:
        """Load Diplomacy-specific templates."""
        return {
            "turn_planning": PromptTemplate(
                system="""You are an expert Diplomacy player. Generate a strategic plan for your turn.

IMPORTANT: You must respond with valid JSON in exactly this format:
{
  "orders": [
    {
      "unit": "A LVP",
      "action": "MTO", 
      "target": "EDI"
    }
  ],
  "rationale": "Brief explanation of your strategy",
  "confidence": 0.8
}

Valid actions:
- HLD: Hold position
- MTO: Move to (target required)
- SUP: Support (support required)
- CVY: Convoy (convoy required)
- RTO: Retreat to (target required)
- DSB: Disband unit

Units are armies (A) or fleets (F) followed by location.""",
                user="""Current game state:
{state}

Your goals:
{goals}

Generate your plan for this turn.""",
                examples=[
                    {
                        "state": "You control: A LVP, F LON, A PAR\nCenters: LVP, LON, PAR",
                        "goals": "Defend your centers, consider expansion",
                        "response": '{"orders": [{"unit": "A LVP", "action": "HLD"}, {"unit": "F LON", "action": "MTO", "target": "ENG"}], "rationale": "Hold LVP, move fleet to English Channel for flexibility", "confidence": 0.7}'
                    }
                ]
            ),
            "plan_repair": PromptTemplate(
                system="""You are an expert Diplomacy player. Repair the given plan based on feedback.

IMPORTANT: You must respond with valid JSON in exactly this format:
{
  "orders": [
    {
      "unit": "A LVP",
      "action": "MTO",
      "target": "EDI"
    }
  ],
  "rationale": "Brief explanation of your strategy",
  "confidence": 0.8
}

Focus on fixing the specific issues mentioned in the feedback.""",
                user="""Current game state:
{state}

Original plan:
{original_plan}

Feedback:
{feedback}

Repair the plan to address the issues.""",
                examples=[]
            )
        }
    
    def build_turn_prompt(
        self,
        state: str,
        goals: Dict[str, Any],
        temperature: float = 0.7
    ) -> str:
        """Build a prompt for turn planning.
        
        Args:
            state: Serialized game state
            goals: Player goals dictionary
            temperature: Sampling temperature
            
        Returns:
            Formatted prompt string
        """
        template = self.templates["turn_planning"]
        
        # Format goals as text
        goals_text = self._format_goals(goals)
        
        # Build examples section
        examples_text = ""
        if template.examples:
            examples_text = "\n\nExamples:\n"
            for i, example in enumerate(template.examples, 1):
                examples_text += f"\nExample {i}:\n"
                examples_text += f"State: {example['state']}\n"
                examples_text += f"Goals: {example['goals']}\n"
                examples_text += f"Response: {example['response']}\n"
        
        # Format user prompt
        user_prompt = template.user.format(
            state=state,
            goals=goals_text
        )
        
        # Combine system and user prompts
        full_prompt = f"{template.system}{examples_text}\n\n{user_prompt}"
        
        return full_prompt
    
    def build_repair_prompt(
        self,
        state: str,
        original_plan: Plan,
        feedback: Dict[str, Any]
    ) -> str:
        """Build a prompt for plan repair.
        
        Args:
            state: Serialized game state
            original_plan: Plan to repair
            feedback: Feedback from simulation
            
        Returns:
            Formatted prompt string
        """
        template = self.templates["plan_repair"]
        
        # Format original plan
        plan_text = self._format_plan(original_plan)
        
        # Format feedback
        feedback_text = self._format_feedback(feedback)
        
        # Format user prompt
        user_prompt = template.user.format(
            state=state,
            original_plan=plan_text,
            feedback=feedback_text
        )
        
        # Combine system and user prompts
        full_prompt = f"{template.system}\n\n{user_prompt}"
        
        return full_prompt
    
    def _format_goals(self, goals: Dict[str, Any]) -> str:
        """Format goals dictionary as text."""
        if not goals:
            return "No specific goals provided"
        
        lines = []
        for key, value in goals.items():
            if isinstance(value, list):
                lines.append(f"- {key}: {', '.join(map(str, value))}")
            else:
                lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)
    
    def _format_plan(self, plan: Plan) -> str:
        """Format plan as text."""
        lines = [f"Rationale: {plan.rationale}", f"Confidence: {plan.confidence}"]
        lines.append("Orders:")
        
        for order_key, order_data in plan.orders.items():
            if isinstance(order_data, dict):
                unit = order_data.get('unit', 'Unknown')
                action = order_data.get('action', 'Unknown')
                target = order_data.get('target', '')
                support = order_data.get('support', '')
                convoy = order_data.get('convoy', '')
                
                order_text = f"  {unit} {action}"
                if target:
                    order_text += f" {target}"
                if support:
                    order_text += f" (supporting {support})"
                if convoy:
                    order_text += f" (convoying {convoy})"
                
                lines.append(order_text)
        
        return "\n".join(lines)
    
    def _format_feedback(self, feedback: Dict[str, Any]) -> str:
        """Format feedback as text."""
        lines = []
        
        if 'is_legal' in feedback:
            lines.append(f"Legal: {feedback['is_legal']}")
        
        if 'score' in feedback:
            lines.append(f"Score: {feedback['score']}")
        
        if 'diagnostics' in feedback:
            diagnostics = feedback['diagnostics']
            if 'illegal_orders' in diagnostics:
                lines.append(f"Illegal orders: {diagnostics['illegal_orders']}")
            if 'bounced_moves' in diagnostics:
                lines.append(f"Bounced moves: {diagnostics['bounced_moves']}")
            if 'errors' in diagnostics:
                lines.append(f"Errors: {diagnostics['errors']}")
        
        if 'suggestions' in feedback:
            lines.append("Suggestions:")
            for suggestion in feedback['suggestions']:
                lines.append(f"  - {suggestion}")
        
        return "\n".join(lines) if lines else "No specific feedback provided"
