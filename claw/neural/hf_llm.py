"""Hugging Face Transformers wrapper for local LLMs."""

import json
import time
from typing import List, Dict, Any, Optional
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig,
    BitsAndBytesConfig
)
from claw.interfaces.neural import NeuralStrategyGenerator
from claw.types import Plan, PlanStatus
from claw.neural.schema import parse_plan_from_json, ValidationError, validate_plan_structure
from claw.neural.prompting import PromptBuilder
from claw.logging import get_logger

logger = get_logger(__name__)


class HuggingFaceLLM(NeuralStrategyGenerator):
    """Hugging Face Transformers wrapper for local LLM inference."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        use_cache: bool = True,
        quantization_config: Optional[BitsAndBytesConfig] = None,
        **kwargs
    ):
        """Initialize the Hugging Face LLM wrapper.
        
        Args:
            model_path: Path to the model or model name
            device: Device to run on ("auto", "cpu", "cuda")
            use_cache: Whether to use model caching
            quantization_config: Optional quantization configuration
            **kwargs: Additional model arguments
        """
        self.model_path = model_path
        self.device = device
        self.use_cache = use_cache
        self.quantization_config = quantization_config
        
        # Initialize model and tokenizer
        self._load_model(**kwargs)
        
        # Initialize prompt builder
        self.prompt_builder = PromptBuilder(domain="diplomacy")
        
        logger.info(f"Initialized HuggingFace LLM with model: {model_path}")
    
    def _load_model(self, **kwargs):
        """Load the model and tokenizer."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            }
            
            if self.quantization_config:
                model_kwargs["quantization_config"] = self.quantization_config
            
            if self.device == "auto":
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = None
                model_kwargs["torch_dtype"] = torch.float32 if self.device == "cpu" else torch.float16
            
            # Add any additional kwargs
            model_kwargs.update(kwargs)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            if self.device != "auto":
                self.model = self.model.to(self.device)
            
            logger.info(f"Model loaded successfully on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_plan(
        self,
        state: Dict[str, Any],
        goals: Dict[str, Any],
        *,
        temperature: float = 0.7,
        max_samples: int = 3
    ) -> List[Plan]:
        """Generate candidate plans using the LLM.
        
        Args:
            state: Current game state
            goals: Player goals and constraints
            temperature: Sampling temperature
            max_samples: Maximum number of plans to generate
            
        Returns:
            List of candidate plans
        """
        logger.info(f"Generating {max_samples} plans with temperature {temperature}")
        
        # Build prompt
        state_str = self._serialize_state(state)
        prompt = self.prompt_builder.build_turn_prompt(state_str, goals, temperature)
        
        # Generate responses
        plans = []
        for i in range(max_samples):
            try:
                plan = self._generate_single_plan(prompt, temperature)
                if plan:
                    plans.append(plan)
                    logger.debug(f"Generated plan {i+1}: {plan.rationale[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to generate plan {i+1}: {e}")
                continue
        
        logger.info(f"Successfully generated {len(plans)} plans")
        return plans
    
    def repair_plan(
        self,
        state: Dict[str, Any],
        plan: Plan,
        feedback: Dict[str, Any],
        *,
        temperature: float = 0.5
    ) -> Plan:
        """Repair a plan based on feedback.
        
        Args:
            state: Current game state
            plan: Plan to repair
            feedback: Feedback from simulation/evaluation
            temperature: Sampling temperature for repair
            
        Returns:
            Repaired plan
        """
        logger.info("Repairing plan based on feedback")
        
        # Build repair prompt
        state_str = self._serialize_state(state)
        prompt = self.prompt_builder.build_repair_prompt(state_str, plan, feedback)
        
        try:
            repaired_plan = self._generate_single_plan(prompt, temperature)
            if repaired_plan:
                repaired_plan.status = PlanStatus.REPAIRED
                logger.info("Plan repaired successfully")
                return repaired_plan
        except Exception as e:
            logger.warning(f"Failed to repair plan: {e}")
        
        # Return original plan if repair fails
        logger.warning("Repair failed, returning original plan")
        return plan
    
    def validate_plan(
        self,
        state: Dict[str, Any],
        plan: Plan
    ) -> bool:
        """Validate a plan for basic correctness.
        
        Args:
            state: Current game state
            plan: Plan to validate
            
        Returns:
            True if plan is valid, False otherwise
        """
        errors = validate_plan_structure(plan)
        if errors:
            logger.debug(f"Plan validation errors: {errors}")
            return False
        return True
    
    def _generate_single_plan(self, prompt: str, temperature: float) -> Optional[Plan]:
        """Generate a single plan from a prompt.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            
        Returns:
            Generated plan or None if generation fails
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            if self.device != "auto":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate configuration
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                max_new_tokens=512,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    use_cache=self.use_cache
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Parse JSON response
            plan = self._parse_response(response)
            return plan
            
        except Exception as e:
            logger.warning(f"Failed to generate plan: {e}")
            return None
    
    def _parse_response(self, response: str) -> Optional[Plan]:
        """Parse LLM response into a Plan object.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed plan or None if parsing fails
        """
        try:
            # Try to extract JSON from response
            json_str = self._extract_json(response)
            if not json_str:
                logger.warning("No JSON found in response")
                return None
            
            # Parse and validate
            plan = parse_plan_from_json(json_str, PlanStatus.PENDING)
            
            # Additional validation
            if not self.validate_plan({}, plan):
                logger.warning("Generated plan failed validation")
                return None
            
            return plan
            
        except ValidationError as e:
            logger.warning(f"Response validation failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to parse response: {e}")
            return None
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from text response.
        
        Args:
            text: Raw text response
            
        Returns:
            JSON string or None if not found
        """
        # Look for JSON block
        start_idx = text.find('{')
        if start_idx == -1:
            return None
        
        # Find matching closing brace
        brace_count = 0
        end_idx = start_idx
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
        
        if brace_count != 0:
            return None
        
        return text[start_idx:end_idx + 1]
    
    def _serialize_state(self, state: Dict[str, Any]) -> str:
        """Serialize state for prompting.
        
        Args:
            state: Game state dictionary
            
        Returns:
            Serialized state string
        """
        # This is a simple implementation - domain adapters should override
        lines = []
        
        if 'units' in state:
            lines.append("Units:")
            for unit, location in state['units'].items():
                lines.append(f"  {unit} in {location}")
        
        if 'centers' in state:
            lines.append("Supply Centers:")
            for power, centers in state['centers'].items():
                lines.append(f"  {power}: {', '.join(centers)}")
        
        if 'phase' in state:
            lines.append(f"Phase: {state['phase']}")
        
        return "\n".join(lines)
